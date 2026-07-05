import { useState } from 'react'
import { identify } from './api'
import type { InferenceResult } from './types'
import { ImageUpload } from './components/ImageUpload'
import { ResultCard } from './components/ResultCard'
import './App.css'

type State =
  | { phase: 'idle' }
  | { phase: 'loading' }
  | { phase: 'result'; result: InferenceResult }
  | { phase: 'error'; message: string }

export function App() {
  const [state, setState] = useState<State>({ phase: 'idle' })

  async function handleSubmit(file: File, lat?: number, lon?: number) {
    setState({ phase: 'loading' })
    try {
      const result = await identify(file, lat, lon)
      setState({ phase: 'result', result })
    } catch (err) {
      setState({ phase: 'error', message: err instanceof Error ? err.message : String(err) })
    }
  }

  function reset() {
    setState({ phase: 'idle' })
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Edible</h1>
        <p>Wild berry identification — educational tool only</p>
      </header>

      <main className="app-main">
        {(state.phase === 'idle' || state.phase === 'loading' || state.phase === 'error') && (
          <ImageUpload onSubmit={handleSubmit} disabled={state.phase === 'loading'} />
        )}

        {state.phase === 'loading' && (
          <div className="loading">
            <div className="spinner" />
            <p>Analyzing image…</p>
          </div>
        )}

        {state.phase === 'error' && (
          <div className="error-box">
            <p>{state.message}</p>
            <button onClick={reset}>Try again</button>
          </div>
        )}

        {state.phase === 'result' && (
          <ResultCard result={state.result} onReset={reset} />
        )}
      </main>
    </div>
  )
}
