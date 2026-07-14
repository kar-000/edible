import { describe, it, expect } from 'vitest'
import { server } from '../test/server'
import { identifyReturns, identifyFails, identifyNetworkError } from '../test/handlers'
import { identify } from '../api'
import { acceptedEdible } from '../test/fixtures'
import { http, HttpResponse } from 'msw'

describe('identify()', () => {
  it('returns parsed JSON on success', async () => {
    server.use(identifyReturns(acceptedEdible))
    const result = await identify(new File(['x'], 'photo.jpg', { type: 'image/jpeg' }))
    expect(result.accepted).toBe(true)
    expect(result.species_common).toBe('Elderberry')
  })

  it('sends a POST to /identify', async () => {
    let capturedRequest: Request | undefined
    server.use(
      http.post('/identify', ({ request }) => {
        capturedRequest = request
        return HttpResponse.json(acceptedEdible)
      })
    )
    await identify(new File(['x'], 'photo.jpg', { type: 'image/jpeg' }))
    expect(capturedRequest?.method).toBe('POST')
    expect(capturedRequest?.url).toContain('/identify')
  })

  it('sends lat and lon as form fields when provided', async () => {
    let body: FormData | undefined
    server.use(
      http.post('/identify', async ({ request }) => {
        body = await request.formData()
        return HttpResponse.json(acceptedEdible)
      })
    )
    await identify(new File(['x'], 'photo.jpg', { type: 'image/jpeg' }), 30.267, -97.743)
    expect(body?.get('lat')).toBe('30.267')
    expect(body?.get('lon')).toBe('-97.743')
  })

  it('omits lat and lon when not provided', async () => {
    let body: FormData | undefined
    server.use(
      http.post('/identify', async ({ request }) => {
        body = await request.formData()
        return HttpResponse.json(acceptedEdible)
      })
    )
    await identify(new File(['x'], 'photo.jpg', { type: 'image/jpeg' }))
    expect(body?.has('lat')).toBe(false)
    expect(body?.has('lon')).toBe(false)
  })

  it('includes the image file in the request', async () => {
    let body: FormData | undefined
    server.use(
      http.post('/identify', async ({ request }) => {
        body = await request.formData()
        return HttpResponse.json(acceptedEdible)
      })
    )
    const file = new File(['content'], 'berry.jpg', { type: 'image/jpeg' })
    await identify(file)
    const sentFile = body?.get('file') as File
    expect(sentFile.name).toBe('berry.jpg')
    expect(sentFile.type).toBe('image/jpeg')
  })

  it('throws on non-200 response', async () => {
    server.use(identifyFails(500, 'Something broke'))
    await expect(
      identify(new File(['x'], 'photo.jpg', { type: 'image/jpeg' }))
    ).rejects.toThrow(/API error 500/)
  })

  it('throws on network error', async () => {
    server.use(identifyNetworkError())
    await expect(
      identify(new File(['x'], 'photo.jpg', { type: 'image/jpeg' }))
    ).rejects.toThrow()
  })
})
