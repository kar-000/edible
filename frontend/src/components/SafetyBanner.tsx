interface Props {
  message: string
  variant: 'danger' | 'warning'
}

export function SafetyBanner({ message, variant }: Props) {
  return (
    <div className={`safety-banner safety-banner--${variant}`}>
      <span className="safety-banner__icon">{variant === 'danger' ? '⛔' : '⚠️'}</span>
      <span>{message}</span>
    </div>
  )
}
