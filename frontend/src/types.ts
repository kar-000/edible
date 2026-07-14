export type Edibility = 'edible_raw' | 'edible_cooked' | 'toxic' | 'uncertain'
export type Severity = 'high' | 'medium' | 'low'
export type RejectionReason = 'not_a_plant' | 'no_fruit_visible' | 'low_confidence' | 'image_invalid'

export interface LookAlikeWarning {
  pair_id: string
  lookalike_common: string
  lookalike_species_id: string
  severity: Severity
  warning_message: string
  distinguishing_features: string[]
  poison_control: string
}

export interface InferenceResult {
  accepted: boolean
  // present when accepted=true
  species_id: string | null
  species_common: string | null
  edibility: Edibility | null
  confidence: number | null
  is_out_of_range: boolean
  lookalike_warnings: LookAlikeWarning[]
  // present when accepted=false
  rejection_reason: RejectionReason | null
  rejection_message: string
  // always present
  disclaimer: string
}

export const CONFIDENCE_FLOOR = 0.75

export function requiresDoNotEatBanner(result: InferenceResult): boolean {
  if (!result.accepted) return true
  if (result.edibility === 'toxic') return true
  if (result.confidence === null || result.confidence < CONFIDENCE_FLOOR) return true
  return false
}
