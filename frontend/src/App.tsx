import { useState } from 'react'
import { identify } from './api'
import type { InferenceResult } from './types'
import { ImageUpload } from './components/ImageUpload'
import { ResultCard } from './components/ResultCard'
import './App.css'

type State =
  | { phase: 'idle' }
  | { phase: 'loading'; imageUrl: string }
  | { phase: 'result'; result: InferenceResult; imageUrl: string }
  | { phase: 'error'; message: string }

export function App() {
  const [state, setState] = useState<State>({ phase: 'idle' })

  async function handleSubmit(file: File, lat?: number, lon?: number) {
    const imageUrl = URL.createObjectURL(file)
    setState({ phase: 'loading', imageUrl })
    try {
      const result = await identify(file, lat, lon)
      setState({ phase: 'result', result, imageUrl })
    } catch (err) {
      URL.revokeObjectURL(imageUrl)
      setState({ phase: 'error', message: err instanceof Error ? err.message : String(err) })
    }
  }

  function reset() {
    if (state.phase === 'result') URL.revokeObjectURL(state.imageUrl)
    setState({ phase: 'idle' })
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Edible</h1>
        <p>Wild berry identification — educational tool only</p>
      </header>

      <main className="app-main">
        {(state.phase === 'idle' || state.phase === 'error') && (
          <ImageUpload onSubmit={handleSubmit} disabled={false} />
        )}

        {state.phase === 'loading' && (
          <div className="loading">
            <img src={state.imageUrl} alt="Analyzing" className="loading-preview" />
            <div className="loading-indicator">
              <div className="spinner" />
              <p>Analyzing image…</p>
            </div>
          </div>
        )}

        {state.phase === 'error' && (
          <div className="error-box">
            <p>{state.message}</p>
            <button onClick={reset}>Try again</button>
          </div>
        )}

        {state.phase === 'result' && (
          <ResultCard result={state.result} imageUrl={state.imageUrl} onReset={reset} />
        )}
      </main>
    </div>
  )
}
