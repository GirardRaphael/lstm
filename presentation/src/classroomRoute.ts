import { SLIDES } from './slides'

export function initialClassroomRoute(hash: string): { mode: 'slides' | 'live' | 'learn'; slideIndex: number } {
  const [view, id] = hash.replace(/^#/, '').split('/')
  return { mode: view === 'slides' || view === 'live' ? view : 'learn',
    slideIndex: Math.max(0, SLIDES.findIndex(slide => slide.id === id)) }
}
