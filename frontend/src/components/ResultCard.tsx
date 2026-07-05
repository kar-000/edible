import type { InferenceResult } from '../types'
import { requiresDoNotEatBanner } from '../types'
import { SafetyBanner } from './SafetyBanner'
import { LookAlikeCard } from './LookAlikeCard'

const EDIBILITY_LABEL: Record<string, string> = {
  edible_raw: 'Edible raw',
  edible_cooked: 'Edible when cooked',
  toxic: 'TOXIC',
  uncertain: 'Uncertain',
}

interface Props {
  result: InferenceResult
  onReset: () => void
}

export function ResultCard({ result, onReset }: Props) {
  const doNotEat = requiresDoNotEatBanner(result)

  return (
    <div className="result-card">
      {!result.accepted ? (
        <>
          <SafetyBanner variant="danger" message={result.rejection_message} />
          <p className="rejection-detail">
            {result.rejection_reason === 'not_a_plant' && 'The image does not appear to contain a plant.'}
            {result.rejection_reason === 'no_fruit_visible' && 'No fruit or berries were visible in the image.'}
            {result.rejection_reason === 'low_confidence' && 'The model was not confident enough to make a safe identification.'}
            {result.rejection_reason === 'image_invalid' && 'The image could not be processed.'}
          </p>
        </>
      ) : (
        <>
          {doNotEat && (
            <SafetyBanner
              variant={result.edibility === 'toxic' ? 'danger' : 'warning'}
              message={result.edibility === 'toxic' ? 'DO NOT EAT — this species is toxic.' : 'Do not eat — confidence too low for a safe identification.'}
            />
          )}

          <div className="result-species">
            <h2>{result.species_common}</h2>
            <p className="species-id">{result.species_id?.replace(/_/g, ' ')}</p>
          </div>

          <div className="result-meta">
            <div className={`edibility-badge edibility-badge--${result.edibility}`}>
              {result.edibility ? EDIBILITY_LABEL[result.edibility] : ''}
            </div>
            <div className="confidence-bar-wrap">
              <div
                className="confidence-bar"
                style={{ width: `${Math.round((result.confidence ?? 0) * 100)}%` }}
              />
              <span className="confidence-label">
                {Math.round((result.confidence ?? 0) * 100)}% confidence
              </span>
            </div>
          </div>

          {result.is_out_of_range && (
            <SafetyBanner
              variant="warning"
              message="This species has not been confirmed in your area. Identification may be less reliable."
            />
          )}

          {result.lookalike_warnings.length > 0 && (
            <div className="lookalikes">
              <h3>Look-alike warnings</h3>
              {result.lookalike_warnings.map((w) => (
                <LookAlikeCard key={w.pair_id} warning={w} />
              ))}
            </div>
          )}
        </>
      )}

      <p className="disclaimer">{result.disclaimer}</p>
      <button onClick={onReset} className="reset-btn">Identify another</button>
    </div>
  )
}
