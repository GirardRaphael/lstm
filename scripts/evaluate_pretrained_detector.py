"""Fixed COCO validation benchmark for the pretrained detector; never trains a policy.

Select 500 images containing a motor vehicle before inference, then measure road
class precision/recall at fixed score=.3 and IoU=.5. Not an intersection benchmark.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import urllib.request
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {1:'person',2:'bicycle',3:'car',4:'motorcycle',6:'bus',8:'truck',10:'traffic light',13:'stop sign'}
SEED = 20261114
SCORE = .3
IOU = .5

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def prepare(root, annotations, count):
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(annotations) as z:
        data=json.loads(z.read('annotations/instances_val2017.json'))
    eligible={a['image_id'] for a in data['annotations'] if a['category_id'] in (3,4,6,8) and not a['iscrowd']}
    ids=sorted(eligible); random.Random(SEED).shuffle(ids); ids=sorted(ids[:count])
    if len(ids)<count: raise ValueError('Not enough eligible vehicle images')
    selected=set(ids)
    manifest={'seed':SEED,'selection':'Random sample of COCO val2017 images with >=1 non-crowd car/motorcycle/bus/truck annotation',
              'images':[i for i in data['images'] if i['id'] in selected],
              'annotations':[{k:a[k] for k in ('id','image_id','category_id','bbox','area','iscrowd')} for a in data['annotations'] if a['image_id'] in selected and a['category_id'] in CLASSES],
              'categories':CLASSES,'score_threshold':SCORE,'iou_threshold':IOU,
              'annotation_zip_sha256':sha(annotations),'licenses':data.get('licenses',[])}
    target=root/'selection.json'
    if target.exists() and json.loads(target.read_text())!=json.loads(json.dumps(manifest)):
        raise ValueError('Refusing to replace the predeclared evaluation sample')
    target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (root/'images').mkdir(exist_ok=True)
    def download(item):
        path=root/'images'/f"{item['id']:012}.jpg"
        if path.exists():
            with Image.open(path) as im: im.verify()
            return
        last=None
        for attempt in range(3):
            for scheme in ('https','http'):
                url=f"{scheme}://images.cocodataset.org/val2017/{item['id']:012}.jpg"
                try:
                    with urllib.request.urlopen(url,timeout=30) as response: raw=response.read(15*1024*1024+1)
                    if len(raw)>15*1024*1024: raise ValueError('Oversized image')
                    partial=path.with_suffix('.part');partial.write_bytes(raw)
                    with Image.open(partial) as im: im.verify()
                    partial.replace(path);return
                except Exception as exc:last=exc
            time.sleep(attempt+1)
        raise RuntimeError(f"Download failed for COCO image {item['id']}: {type(last).__name__}")
    failures=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(download,i) for i in manifest['images']]
        for n,f in enumerate(as_completed(futures),1):
            try:f.result()
            except Exception as exc:failures.append(str(exc))
            if n%50==0 or n==count:print(f'Benchmark images {n}/{count}; failures={len(failures)}',flush=True)
    if failures:raise RuntimeError('\n'.join(failures[:8]))
    hashes={f'{i:012}.jpg':sha(root/'images'/f'{i:012}.jpg') for i in ids}
    hash_manifest=root/'image_hashes.json'
    if hash_manifest.exists() and json.loads(hash_manifest.read_text())!=hashes:
        raise ValueError('Benchmark pixels changed; refusing to replace the original hash manifest')
    if not hash_manifest.exists():hash_manifest.write_text(json.dumps(hashes,indent=2))
    print(json.dumps({'images':len(ids),'ground_truth_objects':len(manifest['annotations']),'class_counts':dict(Counter(CLASSES[a['category_id']] for a in manifest['annotations']))}),flush=True)

def overlap(a,b,crowd=False):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    intersection=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    denominator=aw*ah if crowd else aw*ah+bw*bh-intersection
    return intersection/denominator if denominator>0 else 0.

def match_frame(objects, annotations, width, height):
    result={key:Counter(tp=0,fp=0,fn=0,ignored=0) for key in CLASSES}
    for class_id in CLASSES:
        regular=[a['bbox'] for a in annotations if a['category_id']==class_id and not a['iscrowd']]
        crowds=[a['bbox'] for a in annotations if a['category_id']==class_id and a['iscrowd']]
        unmatched=set(range(len(regular)))
        predictions=sorted((o for o in objects if o['class_id']==class_id and o['score']>=SCORE),key=lambda o:-o['score'])
        for obj in predictions:
            x,y,w,h=obj['box']; box=[x*width,y*height,w*width,h*height]
            candidates=[(overlap(box,regular[i]),i) for i in unmatched]
            best=max(candidates,default=(0,None))
            if best[0]>=IOU:
                result[class_id]['tp']+=1;unmatched.remove(best[1])
            elif any(overlap(box,c,crowd=True)>=IOU for c in crowds):result[class_id]['ignored']+=1
            else:result[class_id]['fp']+=1
        result[class_id]['fn']+=len(unmatched)
    return result

def summarize(counts):
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return {**dict(counts),'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,'ground_truth':tp+fn}

def evaluate(root, model_dir, report):
    sys.path.insert(0,str(ROOT/'presentation/server'))
    from pretrained_detector import PretrainedDetector
    selection=json.loads((root/'selection.json').read_text())
    if selection['score_threshold']!=SCORE or selection['iou_threshold']!=IOU:raise ValueError('Evaluation settings changed')
    detector=PretrainedDetector(model_dir)
    detector_status=detector.status()
    if not detector_status.get('available'):raise RuntimeError(detector_status.get('reason','Detector unavailable'))
    model_hash=sha(model_dir/'efficientdet_lite2.tflite')
    cache_path=root/'predictions.json'
    existing=json.loads(cache_path.read_text()) if cache_path.exists() else {'model_sha256':model_hash,'model_version':detector_status['model_version'],'runtime_version':detector_status['runtime_version'],'image_hashes_sha256':sha(root/'image_hashes.json'),'selection_sha256':sha(root/'selection.json'),'predictions':{}}
    if (existing['model_sha256']!=model_hash or existing['selection_sha256']!=sha(root/'selection.json')
        or existing.get('model_version')!=detector_status['model_version']
        or existing.get('runtime_version')!=detector_status['runtime_version']
        or existing.get('image_hashes_sha256')!=sha(root/'image_hashes.json')):raise ValueError('Prediction cache provenance mismatch')
    image_hashes=json.loads((root/'image_hashes.json').read_text())
    annotation_groups=defaultdict(list)
    for a in selection['annotations']:annotation_groups[a['image_id']].append(a)
    totals={key:Counter(tp=0,fp=0,fn=0,ignored=0) for key in CLASSES}
    latencies=[]
    for n,item in enumerate(selection['images'],1):
        image_id=item['id'];key=str(image_id);path=root/'images'/f'{image_id:012}.jpg'
        if sha(path)!=image_hashes[path.name]:raise ValueError('Benchmark pixels changed')
        if key not in existing['predictions']:
            with Image.open(path) as im: prediction=detector.predict(im.convert('RGB'))
            if not prediction.get('available'):raise RuntimeError(prediction.get('reason','Detector unavailable'))
            existing['predictions'][key]=prediction
            if n%10==0:cache_path.write_text(json.dumps(existing),encoding='utf-8')
        prediction=existing['predictions'][key]
        if prediction.get('model_version')!=detector_status['model_version'] or prediction.get('score_threshold')!=SCORE:
            raise ValueError('Prediction inference contract does not match this benchmark')
        for cat,c in match_frame(prediction['objects'],annotation_groups[image_id],item['width'],item['height']).items():totals[cat].update(c)
        latencies.append(prediction['latency_ms'])
        if n%50==0:print(f'Evaluated {n}/{len(selection["images"])} real vehicle images',flush=True)
    cache_path.write_text(json.dumps(existing),encoding='utf-8')
    aggregate=Counter(tp=0,fp=0,fn=0,ignored=0)
    for c in totals.values():aggregate.update(c)
    per_class={CLASSES[k]:summarize(v) for k,v in totals.items()}
    metrics=summarize(aggregate)
    target_met=len(selection['images'])>=500 and all(c['ground_truth']>=20 and c['precision']>=.9 and c['recall']>=.9 for c in per_class.values())
    result={**metrics,'per_class':per_class,'images':len(selection['images']),'model_sha256':model_hash,'model_version':detector_status['model_version'],'runtime_version':detector_status['runtime_version'],'image_hashes_sha256':sha(root/'image_hashes.json'),'score_threshold':SCORE,'iou_threshold':IOU,
            'scope':'Fixed 500-image vehicle-containing COCO val2017 subset; general real-image recognition, not aerial intersection or signal-decision validation',
            'selection_sha256':sha(root/'selection.json'),'dataset':'COCO val2017','seed':SEED,'evaluated_at':datetime.now(timezone.utc).isoformat(),
            'mean_latency_ms':sum(latencies)/len(latencies),'target':.9,'target_met':target_met,
            'target_definition':'At least 90% precision AND recall for every displayed road class, with >=20 ground-truth instances/class and >=500 images.',
            'scores_calibrated':False,'signal_control_validated':False,'deployment_ready':False,
            'limitations':['Publisher may have used COCO validation while developing the pretrained model; this is a public benchmark, not a fresh private test.',
                           'Vehicle-containing images were sampled; this does not measure false alarms on all-negative scenes.',
                           'COCO images are not a recording-disjoint aerial intersection dataset.','Crowd annotations are excluded from recall; predictions overlapping a same-class crowd at >=0.5 intersection/predicted area are ignored.',
                           'Fixed-threshold precision/recall is not COCO mAP. No detector weights or thresholds were fitted using these evaluation images.']}
    (model_dir/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    rows=['# Pretrained detector: measured real-image benchmark','',f"Model: EfficientDet-Lite2, official Google float32 version1. SHA256 `{model_hash}`. Inference contract `{detector_status['model_version']}`.",'',result['scope']+'.','',f"Score threshold {SCORE}; class-matched IoU threshold {IOU}. Selection seed {SEED}; sample fixed before inference.",'',f"**90% target met: {target_met}.** Precision {metrics['precision']:.2%}; recall {metrics['recall']:.2%}; F1 {metrics['f1']:.2%}.",'','| Class | Ground truth | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    for name,c in per_class.items():rows.append(f"| {name} | {c['ground_truth']} | {c['precision']:.2%} | {c['recall']:.2%} | {c['f1']:.2%} |")
    rows+=['',f"Observed average local inference: {result['mean_latency_ms']:.1f} ms/image. This is not an end-to-end video frame rate.",'',result['target_definition'],'','## Interpretation','',*['- '+x for x in result['limitations']],'','The pretrained detector is used without local weight fine-tuning. Additional aerial examples train the separate anomaly head; reviewed human actions train a separate light-choice model. Those are three distinct learning tasks.','', '## Reproduction','', '`python scripts/evaluate_pretrained_detector.py prepare --annotations <official annotation ZIP>`', '`python scripts/evaluate_pretrained_detector.py evaluate`','', 'Sources: [Google model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector), [COCO](https://cocodataset.org/#download).','']
    report.write_text('\n'.join(rows),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','evaluate']);p.add_argument('--root',type=Path,default=ROOT/'.runtime/detector_benchmark');p.add_argument('--annotations',type=Path,default=ROOT/'.runtime/pretrained_detector/annotations_trainval2017.zip');p.add_argument('--count',type=int,default=500);p.add_argument('--model-dir',type=Path,default=ROOT/'.runtime/pretrained_detector');p.add_argument('--report',type=Path,default=ROOT/'reports/PRETRAINED_DETECTOR_EVALUATION.md');a=p.parse_args()
    if a.stage=='prepare':prepare(a.root,a.annotations,a.count)
    else:evaluate(a.root,a.model_dir,a.report)
