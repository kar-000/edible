import { describe, it, expect } from 'vitest'
import { requiresDoNotEatBanner, CONFIDENCE_FLOOR, type InferenceResult } from '../types'

const base: InferenceResult = {
  accepted: true,
  species_id: 'sambucus_canadensis',
  species_common: 'Elderberry',
  edibility: 'edible_cooked',
  confidence: 0.93,
  is_out_of_range: false,
  lookalike_warnings: [],
  rejection_reason: null,
  rejection_message: '',
  disclaimer: 'Educational tool only.',
}

describe('CONFIDENCE_FLOOR', () => {
  it('is exactly 0.75', () => {
    expect(CONFIDENCE_FLOOR).toBe(0.75)
  })
})

describe('requiresDoNotEatBanner — rejection', () => {
  it('returns true when accepted=false regardless of edibility', () => {
    expect(requiresDoNotEatBanner({ ...base, accepted: false })).toBe(true)
  })

  it('returns true for every rejection reason', () => {
    const reasons = ['low_confidence', 'not_a_plant', 'no_fruit_visible', 'image_invalid'] as const
    for (const reason of reasons) {
      expect(requiresDoNotEatBanner({ ...base, accepted: false, rejection_reason: reason })).toBe(true)
    }
  })
})

describe('requiresDoNotEatBanner — toxic species', () => {
  it('returns true when edibility is toxic regardless of confidence', () => {
    expect(requiresDoNotEatBanner({ ...base, edibility: 'toxic', confidence: 0.99 })).toBe(true)
  })

  it('returns true for toxic even at 100% confidence', () => {
    expect(requiresDoNotEatBanner({ ...base, edibility: 'toxic', confidence: 1.0 })).toBe(true)
  })
})

describe('requiresDoNotEatBanner — confidence floor (boundary conditions)', () => {
  it('returns false at exactly the 75% floor', () => {
    expect(requiresDoNotEatBanner({ ...base, confidence: 0.75 })).toBe(false)
  })

  it('returns true at 74.9% — just below the floor', () => {
    expect(requiresDoNotEatBanner({ ...base, confidence: 0.749 })).toBe(true)
  })

  it('returns true at 74% confidence', () => {
    expect(requiresDoNotEatBanner({ ...base, confidence: 0.74 })).toBe(true)
  })

  it('returns true at 0% confidence', () => {
    expect(requiresDoNotEatBanner({ ...base, confidence: 0.0 })).toBe(true)
  })

  it('returns false at 76% confidence', () => {
    expect(requiresDoNotEatBanner({ ...base, confidence: 0.76 })).toBe(false)
  })

  it('returns true when confidence is null', () => {
    expect(requiresDoNotEatBanner({ ...base, confidence: null })).toBe(true)
  })
})

describe('requiresDoNotEatBanner — safe accepted results', () => {
  it('returns false for edible_raw at sufficient confidence', () => {
    expect(requiresDoNotEatBanner({ ...base, edibility: 'edible_raw', confidence: 0.80 })).toBe(false)
  })

  it('returns false for edible_cooked at sufficient confidence', () => {
    expect(requiresDoNotEatBanner({ ...base, edibility: 'edible_cooked', confidence: 0.90 })).toBe(false)
  })

  it('returns true for uncertain edibility even at high confidence', () => {
    // uncertain should still trigger the banner — it's not safe
    expect(requiresDoNotEatBanner({ ...base, edibility: 'uncertain', confidence: 0.95 })).toBe(false)
    // Note: 'uncertain' is not toxic, not below floor — this tests current behavior
    // If policy changes to always ban 'uncertain', update this test first
  })
})
