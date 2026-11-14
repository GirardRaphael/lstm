import { useEffect, useState } from 'react'
import { SlideView } from './SlideView'
import { SLIDES, type Slide } from './slides'
import './classroom.css'

type LabStatus = { ok?: boolean; detection?: { available?: boolean }; dataset?: { available?: boolean; total?: number }; policy?: { street?: { trained?: boolean; count?: number } } }
type Readiness = { checked: boolean; service: boolean; lab: LabStatus | null }
type Props = { index: number; onIndex: (index: number) => void; onDemo: (demo: NonNullable<Slide['demo']>) => void }

function downloadNotes() {
  const text = SLIDES.map((s, i) => `## ${i + 1}. ${s.title.replace(/\n/g, ' ')}\n\n${s.chapter} · ${s.duration}\n\n**Main point:** ${s.takeaway}\n\n${s.notes.join('\n\n')}\n\n${s.cue ? `**Demonstrate:** ${s.cue}\n\n` : ''}${s.demo ? s.demo.steps.map((step, n) => `${n + 1}. ${step}`).join('\n') : ''}`).join('\n\n---\n\n')
  const url = URL.createObjectURL(new Blob([`# Traffic Intersection — Speaker Notes\n\n${text}`], { type: 'text/markdown;charset=utf-8' }))
  const link = document.createElement('a'); link.href = url; link.download = 'Traffic_Intersection_Speaker_Notes.md'; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function PresentationDeck({ index, onIndex, onDemo }: Props) {
  const [notes, setNotes] = useState(false)
  const [outline, setOutline] = useState(false)
  const [readiness, setReadiness] = useState<Readiness>({ checked: false, service: false, lab: null })
  const [checkVersion, setCheckVersion] = useState(0)
  const [fullscreenMessage, setFullscreenMessage] = useState('')
  const slide = SLIDES[index]
  const chapters = [...new Set(SLIDES.map(s => s.chapter))]
  useEffect(() => {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 8000)
    const get = async (path: string) => { const r = await fetch(path, { signal: controller.signal }); if (!r.ok) throw new Error('Unavailable'); return r.json() }
    let active = true
    void Promise.allSettled([get('/api/health'), get('/api/learning/status')]).then(([health, lab]) => {
      if (active) setReadiness({ checked: true, service: health.status === 'fulfilled' && health.value.ok === true,
        lab: lab.status === 'fulfilled' && lab.value.ok === true ? lab.value as LabStatus : null })
    }).finally(() => clearTimeout(timeout))
    return () => { active = false; controller.abort(); clearTimeout(timeout) }
  }, [checkVersion])
  const fullscreen = async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); setFullscreenMessage('') }
    catch { setFullscreenMessage('Fullscreen is unavailable here. Maximize the browser for presenting.') }
  }
  return <main className="classroom-deck">
    <div className="deck-toolbar">
      <div><p className="visual-eyebrow">CLASSROOM EDITION</p><strong>{slide.chapter}</strong><span className="deck-duration">{slide.duration}{slide.appendix ? ' · Appendix' : ''}</span></div>
      <div className="deck-tools"><button aria-expanded={outline} onClick={() => setOutline(v => !v)}>Slide outline</button><button aria-pressed={notes} onClick={() => setNotes(v => !v)}>{notes ? 'Hide speaker notes' : 'Show speaker notes'}</button><button onClick={() => void fullscreen()}>Fullscreen</button><a href="/classroom/CLASSROOM_PRESENTER_GUIDE.md" download>Presenter guide ↓</a></div>
    </div>
    {fullscreenMessage ? <p role="status">{fullscreenMessage}</p> : null}
    {outline ? <nav className="deck-outline" aria-label="Slide outline">{chapters.map(chapter => <section key={chapter}><h2>{chapter}</h2>{SLIDES.map((s, i) => s.chapter === chapter ? <button key={s.id} aria-current={index === i ? 'step' : undefined} onClick={() => { onIndex(i); setOutline(false) }}><span>{String(i + 1).padStart(2, '0')}</span>{s.title.replace(/\n/g, ' ')}</button> : null)}</section>)}</nav> : null}
    <div className={`deck-layout ${notes ? 'with-notes' : ''}`}>
      <div className="deck-slide-area"><SlideView key={slide.id} slide={slide} index={index} total={SLIDES.length} />
        {slide.demo ? <div className="deck-demo"><div><strong>See it in the app</strong><p>{slide.cue ?? slide.demo.steps[0]}</p></div><button className="accent" onClick={() => onDemo(slide.demo!)}>{slide.demo.label} ↗</button></div> : null}
      </div>
      {notes ? <aside className="speaker-notes" aria-label="Speaker notes"><p className="visual-eyebrow">YOUR TALK TRACK</p><p className="notes-visibility">Visible on this screen. Hide notes before projecting if you want an audience-only view.</p><h2>Explain the idea</h2>{slide.notes.map((note, i) => <p key={i}>{note}</p>)}{slide.demo ? <><h2>Demonstration steps</h2><ol>{slide.demo.steps.map(step => <li key={step}>{step}</li>)}</ol></> : null}{slide.cue ? <div className="speaker-cue"><b>Presenter cue</b><p>{slide.cue}</p></div> : null}{slide.sources?.length ? <><h2>Evidence</h2><ul>{slide.sources.map(source => <li key={source.href}><a href={source.href} target="_blank" rel="noreferrer">{source.label} ↗</a></li>)}</ul></> : null}<button onClick={downloadNotes}>Download all speaker notes</button></aside> : null}
    </div>
    <div className="deck-navigation"><button disabled={index === 0} onClick={() => onIndex(index - 1)}>← Previous</button><label>Slide <select aria-label="Choose slide" value={index} onChange={e => onIndex(Number(e.target.value))}>{SLIDES.map((s, i) => <option key={s.id} value={i}>{i + 1}. {s.title.replace(/\n/g, ' ')}</option>)}</select></label><span>{index + 1} / {SLIDES.length}</span><button className="accent" disabled={index === SLIDES.length - 1} onClick={() => onIndex(index + 1)}>Next →</button></div>
    <div className="deck-progress" role="progressbar" aria-label="Presentation progress" aria-valuemin={1} aria-valuemax={SLIDES.length} aria-valuenow={index + 1}><span style={{ width: `${(index + 1) / SLIDES.length * 100}%` }} /></div>
    <details className="deck-readiness"><summary>Demo readiness · {readiness.checked ? readiness.service ? 'local service responding' : 'local service unavailable' : 'checking…'}</summary><div className="readiness-grid"><p><b>Slides & simulation</b><span>Run in the browser; no model service needed for simulated light control.</span></p><p><b>Real-image detector</b><span>{readiness.lab?.detection?.available ? 'Available locally' : 'Not confirmed — start the service and prepare detector assets.'}</span></p><p><b>Local image gallery</b><span>{readiness.lab?.dataset?.available ? `${readiness.lab.dataset.total?.toLocaleString()} aerial images; roundabout examples do not label signal choices.` : 'Not available — uploads or app screenshots can be used.'}</span></p><p><b>Street decision model</b><span>{readiness.lab ? `${readiness.lab.policy?.street?.count ?? 0} human labels · ${readiness.lab.policy?.street?.trained ? 'trained; inspect its current evaluation' : 'not trained yet'}` : 'Status unavailable; do not claim a trained policy.'}</span></p></div><button onClick={() => { setReadiness(r => ({ ...r, checked: false })); setCheckVersion(v => v + 1) }}>Check again</button><p className="visual-caption">Availability is not a quality test. Evaluation slides record the experiment of 14 November 2026; the learning lab shows current model status.</p></details>
  </main>
}
