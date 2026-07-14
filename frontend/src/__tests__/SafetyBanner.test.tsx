import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { SafetyBanner } from '../components/SafetyBanner'

describe('SafetyBanner', () => {
  it('renders the message', () => {
    render(<SafetyBanner variant="danger" message="DO NOT EAT" />)
    expect(screen.getByText('DO NOT EAT')).toBeInTheDocument()
  })

  it('applies danger class for danger variant', () => {
    const { container } = render(<SafetyBanner variant="danger" message="test" />)
    expect(container.firstChild).toHaveClass('safety-banner--danger')
  })

  it('applies warning class for warning variant', () => {
    const { container } = render(<SafetyBanner variant="warning" message="test" />)
    expect(container.firstChild).toHaveClass('safety-banner--warning')
  })

  it('renders the danger icon for danger variant', () => {
    render(<SafetyBanner variant="danger" message="test" />)
    expect(screen.getByText('⛔')).toBeInTheDocument()
  })

  it('renders the warning icon for warning variant', () => {
    render(<SafetyBanner variant="warning" message="test" />)
    expect(screen.getByText('⚠️')).toBeInTheDocument()
  })
})
