import '@testing-library/jest-dom'
import { afterAll, afterEach, beforeAll } from 'vitest'
import { server } from './server'

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

// jsdom doesn't implement these — stub them out
global.URL.createObjectURL = vi.fn(() => 'blob:mock-image-url')
global.URL.revokeObjectURL = vi.fn()

// Stub geolocation — individual tests override as needed
Object.defineProperty(global.navigator, 'geolocation', {
  value: {
    getCurrentPosition: vi.fn(),
  },
  configurable: true,
  writable: true,
})
