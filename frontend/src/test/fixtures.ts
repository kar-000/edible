import type { InferenceResult, LookAlikeWarning } from '../types'

export const DISCLAIMER = 'Educational tool only. Do not eat wild plants based on this app alone.'

export const lookalike: LookAlikeWarning = {
  pair_id: 'sambucus-phytolacca',
  lookalike_common: 'Pokeweed',
  lookalike_species_id: 'phytolacca_americana',
  severity: 'high',
  warning_message: 'Pokeweed berries are highly toxic. Clusters look similar but grow on thick magenta stems.',
  distinguishing_features: [
    'Elderberry stems are woody and grey; pokeweed has thick magenta-red stems',
    'Elderberry leaflets are serrated in pairs; pokeweed leaves are large and undivided',
  ],
  poison_control: '1-800-222-1222',
}

export const acceptedEdible: InferenceResult = {
  accepted: true,
  species_id: 'sambucus_canadensis',
  species_common: 'Elderberry',
  edibility: 'edible_cooked',
  confidence: 0.93,
  is_out_of_range: false,
  lookalike_warnings: [],
  rejection_reason: null,
  rejection_message: '',
  disclaimer: DISCLAIMER,
}

export const acceptedEdibleHighConf: InferenceResult = {
  ...acceptedEdible,
  confidence: 0.97,
}

export const acceptedEdibleMidConf: InferenceResult = {
  ...acceptedEdible,
  confidence: 0.78,
}

export const acceptedEdibleAtFloor: InferenceResult = {
  ...acceptedEdible,
  confidence: 0.75,
}

export const acceptedEdibleBelowFloor: InferenceResult = {
  ...acceptedEdible,
  confidence: 0.749,
}

export const acceptedToxic: InferenceResult = {
  accepted: true,
  species_id: 'phytolacca_americana',
  species_common: 'Pokeweed',
  edibility: 'toxic',
  confidence: 0.88,
  is_out_of_range: false,
  lookalike_warnings: [],
  rejection_reason: null,
  rejection_message: '',
  disclaimer: DISCLAIMER,
}

export const rejectedLowConf: InferenceResult = {
  accepted: false,
  species_id: null,
  species_common: null,
  edibility: null,
  confidence: null,
  is_out_of_range: false,
  lookalike_warnings: [],
  rejection_reason: 'low_confidence',
  rejection_message: 'Could not make a safe identification. Do not eat.',
  disclaimer: DISCLAIMER,
}

export const rejectedNotPlant: InferenceResult = {
  ...rejectedLowConf,
  rejection_reason: 'not_a_plant',
  rejection_message: 'Could not identify — no plant detected.',
}

export const rejectedNoFruit: InferenceResult = {
  ...rejectedLowConf,
  rejection_reason: 'no_fruit_visible',
  rejection_message: 'Could not identify — fruit not visible.',
}

export const acceptedEdibleOutOfRange: InferenceResult = {
  ...acceptedEdible,
  is_out_of_range: true,
}

export const rejectedOutOfRange: InferenceResult = {
  ...rejectedLowConf,
  is_out_of_range: true,
}

export const acceptedEdibleWithLookalike: InferenceResult = {
  ...acceptedEdible,
  lookalike_warnings: [lookalike],
}
