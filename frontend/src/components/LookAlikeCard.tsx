import type { LookAlikeWarning } from '../types'

interface Props {
  warning: LookAlikeWarning
}

export function LookAlikeCard({ warning }: Props) {
  return (
    <div className={`lookalike-card lookalike-card--${warning.severity}`}>
      <div className="lookalike-card__header">
        <strong>Look-alike: {warning.lookalike_common}</strong>
        <span className={`severity-badge severity-badge--${warning.severity}`}>
          {warning.severity.toUpperCase()}
        </span>
      </div>
      <p className="lookalike-card__warning">{warning.warning_message}</p>
      <ul className="lookalike-card__features">
        {warning.distinguishing_features.map((f, i) => (
          <li key={i}>{f}</li>
        ))}
      </ul>
      <p className="lookalike-card__poison">
        Poison Control: <a href={`tel:${warning.poison_control}`}>{warning.poison_control}</a>
      </p>
    </div>
  )
}
