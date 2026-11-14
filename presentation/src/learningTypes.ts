export type Action = 'ns' | 'ew' | 'walk' | 'hold'
export type Domain = 'street' | 'simulation'
export type Source = 'dataset' | 'upload' | 'simulation'
export type Metrics = Record<string, unknown>
export type Detection = {
  available: boolean; model?: string; model_version?: string; pretrained_on?: string; telemetry_notice?: string
  objects?: { class_id: number; class_name: string; score: number; box: [number, number, number, number] }[]
  counts?: Record<string, number>; score_threshold?: number; scores_calibrated?: boolean
  latency_ms?: number; image_size?: [number, number]; target?: number; target_met?: boolean
  evaluation?: { precision: number; recall: number; f1: number; images: number; iou_threshold: number; scope: string } | null
  reason?: string
}
export type PolicyStatus = {
  trained: boolean; count: number; group_count: number; label_counts: Partial<Record<Action, number>>
  requirements?: { min_samples: number; min_groups: number; min_per_action: number }
  reason?: string; model_version?: string | null; metrics?: Metrics | null; calibrated?: boolean; evaluation_reused?: boolean
  reliability_target?: number; confidence_threshold?: number; target_met?: boolean | null; target_reason?: string; interval_limitation?: string
}
export type Status = {
  ok: boolean
  detection?: Detection
  vision: { available: boolean; model_version?: string; metrics?: Metrics | null; reason?: string; calibrated?: boolean; deployment_ready?: boolean; validation_passed?: boolean; adequate?: boolean; evaluation_reused?: boolean; validation_result?: unknown }
  policy: Partial<Record<Domain, PolicyStatus>>
  dataset: { available: boolean; total: number; counts?: Record<string, number> }
}
export type Sample = { sample_id: string; source_video: string; split: string; label: number | string; class_name: string; image_url: string }
export type Samples = { ok: boolean; samples: Sample[]; total: number }
export type AnomalyJudgement = 'normal' | 'anomaly' | 'uncertain'
export type AnomalyReview = { judgement: AnomalyJudgement; label: 0 | 1 | null; reason: string; reviewed_at: string; label_source: 'human' }
export type Analysis = {
  ok: boolean; snapshot_id: string; video?: VideoMetadata; domain: Domain; group_id: string; decision_teaching_allowed?: boolean; image_url?: string; detection?: Detection
  anomaly?: { anomaly_probability?: number | null; calibrated?: boolean; decision?: string; model_version?: string; reason?: string; [key: string]: unknown }
  anomaly_review?: AnomalyReview | null
  policy: { trained?: boolean; action?: Action | null; suggested_action?: Action | null; probabilities?: Partial<Record<Action, number>> | null; confidence?: number | null; calibrated?: boolean; abstain?: boolean; reason?: string; model_version?: string | null; confidence_threshold?: number; reliability_target?: number; target_met?: boolean | null; target_reason?: string }
}
export type Candidate = { preview: string; title: string; domain: Domain; groupId: string; sampleId?: string; dataUrl?: string; video?: VideoMetadata }
export type VideoMetadata = { recording_id: string; name: string; timestamp_seconds: number; source_url?: string; license?: string }
export type VideoAnalysis = Omit<Analysis, 'snapshot_id'> & { snapshot_id?: string }
