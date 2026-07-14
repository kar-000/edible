import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect } from 'vitest'
import { App } from '../App'
import { server } from '../test/server'
import {
  identifyReturns,
  identifyFails,
  identifyNetworkError,
  identifyHangs,
} from '../test/handlers'
import {
  acceptedEdible,
  rejectedLowConf,
  acceptedToxic,
} from '../test/fixtures'

function makeFile() {
  return new File(['berry'], 'berry.jpg', { type: 'image/jpeg' })
}

async function uploadAndSubmit(user: ReturnType<typeof userEvent.setup>, container: HTMLElement) {
  const input = container.querySelector('input[type="file"]') as HTMLInputElement
  await user.upload(input, makeFile())
  await user.click(container.querySelector('button[type="submit"]') as HTMLButtonElement)
}

// ─── Page navigation ──────────────────────────────────────────────

describe('navigation', () => {
  it('shows the identify page by default', () => {
    render(<App />)
    expect(screen.getByText(/drop a berry photo/i)).toBeInTheDocument()
  })

  it('shows the disclaimer banner on the identify page', () => {
    render(<App />)
    expect(screen.getByText(/educational tool — not a foraging guide/i)).toBeInTheDocument()
  })

  it('navigates to the About page', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: /about/i }))
    expect(screen.getByText(/how identification works/i)).toBeInTheDocument()
  })

  it('navigates back to Identify from About', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: /about/i }))
    await user.click(screen.getByRole('button', { name: /^identify$/i }))
    expect(screen.getByText(/drop a berry photo/i)).toBeInTheDocument()
  })

  it('clicking the brand name returns to Identify page', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: /about/i }))
    await user.click(screen.getByRole('button', { name: /^edible$/i }))
    expect(screen.getByText(/drop a berry photo/i)).toBeInTheDocument()
  })

  it('disclaimer banner is not shown on About page', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: /about/i }))
    expect(screen.queryByText(/educational tool — not a foraging guide/i)).not.toBeInTheDocument()
  })
})

// ─── Upload → loading → result ────────────────────────────────────

describe('identification flow — success', () => {
  it('shows loading state while request is in flight', async () => {
    server.use(identifyHangs())
    const user = userEvent.setup()
    const { container } = render(<App />)
    await uploadAndSubmit(user, container)
    await waitFor(() => {
      expect(screen.getByAltText(/analyzing/i)).toBeInTheDocument()
    })
  })

  it('shows the result card after a successful identification', async () => {
    server.use(identifyReturns(acceptedEdible))
    const user = userEvent.setup()
    const { container } = render(<App />)
    await uploadAndSubmit(user, container)
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /elderberry/i })).toBeInTheDocument()
    })
  })

  it('shows a danger banner for a toxic identification', async () => {
    server.use(identifyReturns(acceptedToxic))
    const user = userEvent.setup()
    const { container } = render(<App />)
    await uploadAndSubmit(user, container)
    await waitFor(() => {
      expect(screen.getByText(/DO NOT EAT/)).toBeInTheDocument()
    })
  })

  it('shows rejection message for a rejected result', async () => {
    server.use(identifyReturns(rejectedLowConf))
    const user = userEvent.setup()
    const { container } = render(<App />)
    await uploadAndSubmit(user, container)
    await waitFor(() => {
      expect(screen.getByText(/could not make a safe identification/i)).toBeInTheDocument()
    })
  })

  it('"Identify another" resets to idle state', async () => {
    server.use(identifyReturns(acceptedEdible))
    const user = userEvent.setup()
    const { container } = render(<App />)
    await uploadAndSubmit(user, container)
    await waitFor(() => screen.getByRole('button', { name: /identify another/i }))
    await user.click(screen.getByRole('button', { name: /identify another/i }))
    expect(screen.getByText(/drop a berry photo/i)).toBeInTheDocument()
  })
})

// ─── Error handling ───────────────────────────────────────────────

describe('identification flow — errors', () => {
  it('shows an error message on API failure', async () => {
    server.use(identifyFails(500, 'Internal server error'))
    const user = userEvent.setup()
    const { container } = render(<App />)
    await uploadAndSubmit(user, container)
    await waitFor(() => {
      expect(screen.getByText(/API error 500/i)).toBeInTheDocument()
    })
  })

  it('shows an error message on network failure', async () => {
    server.use(identifyNetworkError())
    const user = userEvent.setup()
    const { container } = render(<App />)
    await uploadAndSubmit(user, container)
    await waitFor(() => {
      expect(screen.getByText(/failed to fetch/i)).toBeInTheDocument()
    })
  })

  it('"Try again" after error restores upload form', async () => {
    server.use(identifyFails(500))
    const user = userEvent.setup()
    const { container } = render(<App />)
    await uploadAndSubmit(user, container)
    await waitFor(() => screen.getByRole('button', { name: /try again/i }))
    await user.click(screen.getByRole('button', { name: /try again/i }))
    expect(screen.getByText(/drop a berry photo/i)).toBeInTheDocument()
  })
})
