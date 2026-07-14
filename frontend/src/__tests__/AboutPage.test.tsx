import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { AboutPage } from '../components/AboutPage'

describe('AboutPage', () => {
  it('explains the 3-layer pipeline', () => {
    render(<AboutPage />)
    // getAllByText because these labels appear in both the pipeline steps and data sources
    expect(screen.getAllByText(/plant gate/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/fruit presence gate/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/species classifier/i).length).toBeGreaterThan(0)
  })

  it('shows the 75% confidence floor', () => {
    render(<AboutPage />)
    expect(screen.getAllByText(/75/).length).toBeGreaterThan(0)
  })

  it('explains GPS range checking', () => {
    render(<AboutPage />)
    expect(screen.getByText(/range checking/i)).toBeInTheDocument()
  })

  it('lists all 6 edible species', () => {
    render(<AboutPage />)
    const edibleNames = ['Beautyberry', 'Sugarberry', 'Agarita', 'Dewberry', 'Elderberry', 'Mustang grape']
    for (const name of edibleNames) {
      expect(screen.getAllByText(new RegExp(name, 'i')).length).toBeGreaterThan(0)
    }
  })

  it('lists all 6 toxic species', () => {
    render(<AboutPage />)
    const toxicNames = ['Possumhaw', 'Yaupon holly', 'Chinaberry', 'Pokeweed', 'Black nightshade', 'Carolina horsenettle']
    for (const name of toxicNames) {
      expect(screen.getAllByText(new RegExp(name, 'i')).length).toBeGreaterThan(0)
    }
  })

  it('attributes iNaturalist as a data source', () => {
    render(<AboutPage />)
    expect(screen.getAllByText(/iNaturalist/i).length).toBeGreaterThan(0)
  })

  it('attributes USDA PLANTS as a data source', () => {
    render(<AboutPage />)
    expect(screen.getAllByText(/USDA PLANTS/i).length).toBeGreaterThan(0)
  })

  it('attributes OpenStreetMap Nominatim for geocoding', () => {
    render(<AboutPage />)
    expect(screen.getAllByText(/Nominatim/i).length).toBeGreaterThan(0)
  })

  it('attributes DINOv2 as the model backbone', () => {
    render(<AboutPage />)
    expect(screen.getAllByText(/DINOv2/i).length).toBeGreaterThan(0)
  })

  it('shows the educational disclaimer', () => {
    render(<AboutPage />)
    expect(screen.getAllByText(/educational tool only/i).length).toBeGreaterThan(0)
  })

  it('warns that misidentification can cause illness or death', () => {
    render(<AboutPage />)
    expect(screen.getByText(/serious illness or death/i)).toBeInTheDocument()
  })
})
