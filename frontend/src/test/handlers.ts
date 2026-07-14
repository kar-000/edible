import { delay, http, HttpResponse } from 'msw'
import { acceptedEdible, DISCLAIMER } from './fixtures'

export const handlers = [
  http.post('/identify', () => HttpResponse.json(acceptedEdible)),
]

// Convenience factories for per-test overrides
export const identifyReturns = (body: object) =>
  http.post('/identify', () => HttpResponse.json(body))

export const identifyFails = (status = 500, message = 'Internal server error') =>
  http.post('/identify', () => HttpResponse.json({ detail: message }, { status }))

export const identifyNetworkError = () =>
  http.post('/identify', () => HttpResponse.error())

export const identifyHangs = () =>
  http.post('/identify', async () => {
    await delay('infinite')
    return HttpResponse.json({})
  })

export const nominatimReturns = (lat: string, lon: string) =>
  http.get('https://nominatim.openstreetmap.org/search', () =>
    HttpResponse.json([{ lat, lon, display_name: 'Test City, TX, US' }])
  )

export const nominatimNotFound = () =>
  http.get('https://nominatim.openstreetmap.org/search', () =>
    HttpResponse.json([])
  )

export const nominatimFails = () =>
  http.get('https://nominatim.openstreetmap.org/search', () =>
    HttpResponse.error()
  )
