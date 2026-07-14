import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert, Box, Button, Chip, CircularProgress, Collapse, IconButton, Slider, Stack, TextField, Tooltip, Typography,
} from '@mui/material'
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate'
import CloseIcon from '@mui/icons-material/Close'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import StarIcon from '@mui/icons-material/Star'
import StarBorderIcon from '@mui/icons-material/StarBorder'
import { useStore } from '../../store'
import { apiFetch, publicUrl, type Mood, type MoodboardItem, type MoodboardSuggestion } from '../../api'

type CatalogSource = 'official' | 'andrometa' | 'favorites' | 'suggested'

function moodboardPreviews(board: MoodboardItem): string[] {
  const images = board.preview_image_urls?.length
    ? board.preview_image_urls
    : board.image_urls?.length ? board.image_urls : [board.primary_image_url].filter(Boolean)
  return images.slice(0, 4)
}

function moodboardImageSrc(src: string): string {
  if (/^(?:https?:|data:|blob:)/i.test(src)) return src
  return publicUrl(src)
}

function moodboardErrorMessage(error: any, fallback: string) {
  const detail = error?.response?.data?.detail
  if (detail === 'Authentication required') {
    return 'Sign in to use shared moodboard actions, or run local mode for unauthenticated access.'
  }
  if (detail === 'Admin access required') {
    return 'Admin login is required for this moodboard action in sharing mode.'
  }
  return detail ?? error?.message ?? fallback
}

interface MoodboardSectionProps {
  intro?: string
  promptValue?: string
  onPromptFallback?: (prompt: string) => void
  applyTitleToPrompt?: boolean
}

export default function MoodboardSection({
  intro = '',
  promptValue,
  onPromptFallback,
  applyTitleToPrompt = true,
}: MoodboardSectionProps) {
  const { params, setParam, moodboardSuggestions } = useStore()
  const [moods, setMoods] = useState<Mood[]>([])
  const [catalogQuery, setCatalogQuery] = useState('')
  const [catalogSource, setCatalogSource] = useState<CatalogSource>('official')
  const [catalogResults, setCatalogResults] = useState<MoodboardItem[]>([])
  const [catalogPage, setCatalogPage] = useState(1)
  const [catalogTotal, setCatalogTotal] = useState(0)
  const [selectedBoards, setSelectedBoards] = useState<MoodboardItem[]>([])
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [mashupLoading, setMashupLoading] = useState(false)
  const [catalogMessage, setCatalogMessage] = useState('')
  const [open, setOpen] = useState(true)
  const fileRef = useRef<HTMLInputElement>(null)
  const catalogScrollRef = useRef<HTMLDivElement>(null)
  const catalogSentinelRef = useRef<HTMLDivElement>(null)
  const catalogLoadingRef = useRef(false)

  useEffect(() => { apiFetch.moods().then(setMoods).catch(() => {}) }, [])

  const board = params.moodboard_images
  const active = !!params.mood || params.selected_moodboard_ids.length > 0 || board.length > 0

  const selectedIds = params.mood.split(',').map(id => id.trim()).filter(Boolean)
  const selectedCatalogIds = params.selected_moodboard_ids
  const selectedMoods = selectedIds
    .map(id => moods.find(m => m.id === id))
    .filter((m): m is Mood => !!m)

  const pickMood = (id: string) => {
    const next = selectedIds.includes(id)
      ? selectedIds.filter(existing => existing !== id)
      : [...selectedIds, id]
    setParam('mood', next.join(','))
  }

  useEffect(() => {
    const missing = selectedCatalogIds.filter(id => !selectedBoards.some(board => board.id === id))
    if (!missing.length) return
    Promise.all(missing.map(id => apiFetch.moodboard(id).catch(() => null)))
      .then(items => setSelectedBoards(prev => [
        ...prev,
        ...items.filter((item): item is MoodboardItem => !!item && !prev.some(board => board.id === item.id)),
      ]))
      .catch(() => undefined)
  }, [selectedCatalogIds, selectedBoards])

  const CATALOG_PAGE_SIZE = 60

  const catalogQueryRef = useRef(catalogQuery)
  catalogQueryRef.current = catalogQuery
  const catalogSourceRef = useRef(catalogSource)
  catalogSourceRef.current = catalogSource
  const catalogPageRef = useRef(catalogPage)
  catalogPageRef.current = catalogPage
  const catalogTotalRef = useRef(catalogTotal)
  catalogTotalRef.current = catalogTotal
  const catalogResultsLenRef = useRef(catalogResults.length)
  catalogResultsLenRef.current = catalogResults.length

  const searchCatalog = useCallback(async (query = catalogQueryRef.current, src = catalogSourceRef.current, page = 1, append = false) => {
    if (src === 'suggested') {
      setCatalogMessage(moodboardSuggestions.length ? '' : 'Use Magic Wand to suggest moodboards for your prompt.')
      return
    }
    if (append) {
      if (catalogLoadingRef.current) return
      if (catalogResultsLenRef.current >= catalogTotalRef.current) return
    }
    catalogLoadingRef.current = true
    setCatalogLoading(true)
    if (!append) setCatalogMessage('')
    try {
      const opts: { q: string; page: number; pageSize: number; favorites?: boolean; source?: 'official' | 'andrometa' } =
        { q: query, page, pageSize: CATALOG_PAGE_SIZE }
      if (src === 'favorites') opts.favorites = true
      else opts.source = src
      const data = await apiFetch.moodboards(opts)
      setCatalogResults(prev => {
        if (!append) return data.items
        const seen = new Set(prev.map(item => item.id))
        return [...prev, ...data.items.filter(item => !seen.has(item.id))]
      })
      setCatalogTotal(data.total)
      setCatalogPage(page)
      if (!data.total) {
        setCatalogMessage(src === 'favorites'
          ? 'No favorited moodboards yet — tap the ☆ on any board to save it here.'
          : 'No moodboards matched that search.')
      } else {
        setCatalogMessage('')
      }
    } catch (e: any) {
      setCatalogMessage(moodboardErrorMessage(e, 'Could not search moodboards.'))
    } finally {
      catalogLoadingRef.current = false
      setCatalogLoading(false)
    }
  }, [moodboardSuggestions.length])

  const selectSource = (src: CatalogSource) => {
    setCatalogSource(src)
    void searchCatalog(catalogQuery, src, 1, false)
  }

  useEffect(() => {
    if (moodboardSuggestions.length > 0) {
      setOpen(true)
      setCatalogSource('suggested')
    }
  }, [moodboardSuggestions.length])

  const toggleCatalogFavorite = async (item: MoodboardItem) => {
    const next = !item.favorite
    try {
      await apiFetch.setMoodboardFavorite(item.id, next)
      if (catalogSource === 'favorites' && !next) {
        setCatalogResults(prev => prev.filter(r => r.id !== item.id))
        setCatalogTotal(t => Math.max(0, t - 1))
      } else {
        setCatalogResults(prev => prev.map(r => r.id === item.id ? { ...r, favorite: next } : r))
      }
    } catch (e: any) {
      setCatalogMessage(moodboardErrorMessage(e, 'Could not update favorite.'))
    }
  }

  useEffect(() => {
    if (!open || catalogResults.length) return
    void searchCatalog('', catalogSource)
    // Only auto-load a small browse set when the section is first opened.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // Infinite scroll inside the catalog list panel.
  useEffect(() => {
    if (catalogSource === 'suggested') return
    const root = catalogScrollRef.current
    const sentinel = catalogSentinelRef.current
    if (!root || !sentinel) return
    const observer = new IntersectionObserver((entries) => {
      if (!entries.some(entry => entry.isIntersecting)) return
      if (catalogLoadingRef.current) return
      if (catalogResultsLenRef.current <= 0) return
      if (catalogResultsLenRef.current >= catalogTotalRef.current) return
      void searchCatalog(
        catalogQueryRef.current,
        catalogSourceRef.current,
        catalogPageRef.current + 1,
        true,
      )
    }, { root, rootMargin: '240px', threshold: 0 })
    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [catalogSource, catalogResults.length, catalogTotal, catalogLoading, searchCatalog])

  const addCatalogMoodboard = async (moodboard: MoodboardItem) => {
    setCatalogLoading(true)
    setCatalogMessage('')
    try {
      const current = useStore.getState().params
      const nextIds = Array.from(new Set([...current.selected_moodboard_ids, moodboard.id]))
      const nextUuids = Array.from(new Set([...current.moodboard_uuids, moodboard.uuid].filter(Boolean)))
      setSelectedBoards(prev => prev.some(board => board.id === moodboard.id) ? prev : [...prev, moodboard])
      setParam('selected_moodboard_ids', nextIds)
      setParam('moodboard_uuids', nextUuids)
      const existingPrompt = promptValue ?? current.prompt
      if (applyTitleToPrompt && !existingPrompt.trim()) {
        if (onPromptFallback) onPromptFallback(moodboard.title)
        else setParam('prompt', moodboard.title)
      }
    } catch (e: any) {
      setCatalogMessage(moodboardErrorMessage(e, 'Could not add Krea moodboard.'))
    } finally {
      setCatalogLoading(false)
    }
  }

  const addSuggestedMoodboard = (moodboard: MoodboardSuggestion) => {
    const nextIds = Array.from(new Set([...params.selected_moodboard_ids, moodboard.id]))
    const nextUuids = Array.from(new Set([
      ...params.moodboard_uuids,
      ...(moodboard.uuid ? [moodboard.uuid] : []),
    ]))
    setParam('selected_moodboard_ids', nextIds)
    setParam('moodboard_uuids', nextUuids)
  }

  const removeCatalogMoodboard = (id: number) => {
    const board = selectedBoards.find(board => board.id === id)
    setParam('selected_moodboard_ids', selectedCatalogIds.filter(existing => existing !== id))
    if (board?.uuid) setParam('moodboard_uuids', params.moodboard_uuids.filter(uuid => uuid !== board.uuid))
    setSelectedBoards(prev => prev.filter(board => board.id !== id))
  }

  const createMashupFromSelected = async () => {
    if (selectedCatalogIds.length < 2) {
      setCatalogMessage('Select at least two Krea catalog moodboards to create a mashup.')
      return
    }
    setMashupLoading(true)
    setCatalogMessage('')
    try {
      const created = await apiFetch.createMoodboardMashup({
        moodboard_ids: selectedCatalogIds,
        weights: selectedCatalogIds.map(() => 1.0),
      })
      setSelectedBoards([created])
      setParam('selected_moodboard_ids', [created.id])
      setParam('moodboard_uuids', created.uuid ? [created.uuid] : [])
      setParam('moodboard_strength', 0.35)
      setCatalogMessage(`Created mashup moodboard "${created.title}" and applied it.`)
    } catch (e: any) {
      setCatalogMessage(moodboardErrorMessage(e, 'Could not create moodboard mashup.'))
    } finally {
      setMashupLoading(false)
    }
  }

  const addImages = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    files.forEach(f => {
      const r = new FileReader()
      r.onload = ev => {
        const b64 = (ev.target?.result as string).split(',')[1]
        setParam('moodboard_images', [...useStore.getState().params.moodboard_images, b64])
      }
      r.readAsDataURL(f)
    })
    e.target.value = ''
  }

  const removeImage = (i: number) =>
    setParam('moodboard_images', board.filter((_, idx) => idx !== i))

  const summary = selectedMoods.length
    ? selectedMoods.map(m => m.name).join(' + ')
    : ''
  const catalogSummary = selectedCatalogIds.length ? `${selectedCatalogIds.length} Krea board${selectedCatalogIds.length === 1 ? '' : 's'}` : ''

  return (
    <Box>
      <Stack direction="row" alignItems="center" justifyContent="space-between"
        sx={{ cursor: 'pointer' }} onClick={() => setOpen(o => !o)}>
        <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
          Moodboard{active ? ` · ${[summary, catalogSummary, board.length ? `${board.length} img` : ''].filter(Boolean).join(' + ')}` : ''}
          <Tooltip title="Moodboards are style controls. Catalog moodboards apply Qwen-enriched text guidance by default; uploaded images are optional stronger visual references.">
            <InfoOutlinedIcon sx={{ fontSize: 14, color: 'text.disabled', ml: 0.5, verticalAlign: 'middle' }} />
          </Tooltip>
        </Typography>
        <ExpandMoreIcon sx={{ color: 'text.secondary', transform: open ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
      </Stack>

      <Collapse in={open}>
        <Box sx={{ pt: 1 }}>
          {intro && (
            <Alert severity="info" sx={{ py: 0.75, mb: 1.5 }}>
              {intro}
            </Alert>
          )}

          {selectedMoods.length > 0 && (
            <Box sx={{ mb: 1.5 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, fontWeight: 600 }}>
                Style Stack
              </Typography>
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {selectedMoods.map(m => (
                  <Chip
                    key={m.id}
                    label={`${m.emoji} ${m.name}`}
                    size="small"
                    color={m.category === 'Horror' ? 'error' : 'secondary'}
                    onDelete={() => pickMood(m.id)}
                  />
                ))}
              </Stack>
            </Box>
          )}

          {selectedCatalogIds.length > 0 && (
            <Box sx={{ mb: 1.5 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, fontWeight: 600 }}>
                Krea Catalog Moodboards
              </Typography>
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {selectedCatalogIds.map(id => {
                  const selected = selectedBoards.find(board => board.id === id)
                  return (
                    <Chip
                      key={id}
                      label={selected?.title ?? `Moodboard #${id}`}
                      size="small"
                      color="primary"
                      onDelete={() => removeCatalogMoodboard(id)}
                    />
                  )
                })}
              </Stack>
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
                Catalog moodboards use their enriched Qwen style guidance by default. Add reference images below only when you want a stronger visual pull.
              </Typography>
              {selectedCatalogIds.length >= 2 && (
                <Button
                  variant="outlined"
                  size="small"
                  sx={{ mt: 1 }}
                  disabled={mashupLoading}
                  onClick={createMashupFromSelected}
                >
                  {mashupLoading ? 'Creating mashup...' : `Create mashup from selected (${selectedCatalogIds.length})`}
                </Button>
              )}
            </Box>
          )}

          {/* Krea catalog moodboards */}
          <Box sx={{ mb: 1.5 }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontWeight: 600 }}>
                Browse Moodboards
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                Click Add to apply · ☆ to favorite · 2+ to mash up.
              </Typography>
            </Stack>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap sx={{ mb: 0.75 }}>
              {([
                ['suggested', `Suggested${moodboardSuggestions.length ? ` (${moodboardSuggestions.length})` : ''}`],
                ['official', 'Official Krea'],
                ['favorites', '★ Favorites'],
                ['andrometa', 'Andro.Meta'],
              ] as const).map(([src, label]) => (
                <Chip
                  key={src}
                  size="small"
                  clickable
                  label={label}
                  variant={catalogSource === src ? 'filled' : 'outlined'}
                  color={catalogSource === src ? 'primary' : 'default'}
                  onClick={() => selectSource(src)}
                />
              ))}
            </Stack>
            {catalogSource !== 'suggested' && <Stack direction={{ xs: 'column', sm: 'row' }} spacing={0.75}>
              <TextField
                size="small"
                value={catalogQuery}
                onChange={e => setCatalogQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') searchCatalog(catalogQuery) }}
                placeholder="fantasy, sci-fi, product, noir, ethereal..."
                fullWidth
              />
              <Button variant="outlined" onClick={() => searchCatalog(catalogQuery)} disabled={catalogLoading}>
                {catalogLoading ? <CircularProgress size={16} /> : 'Search'}
              </Button>
            </Stack>}
            {catalogSource !== 'suggested' && <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>
              {['fantasy', 'sci-fi', 'product', 'noir', 'ethereal'].map(term => (
                <Chip
                  key={term}
                  size="small"
                  clickable
                  label={term}
                  variant="outlined"
                  onClick={() => {
                    setCatalogQuery(term)
                    searchCatalog(term)
                  }}
                />
              ))}
            </Stack>}
            {catalogMessage && (
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
                {catalogMessage}
              </Typography>
            )}
            {catalogSource === 'suggested' && moodboardSuggestions.length > 0 && (
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
                Suggested by Magic Wand for your current prompt. Add one or more to apply their text guidance.
              </Typography>
            )}
            {((catalogSource === 'suggested' ? moodboardSuggestions : catalogResults).length > 0) && (
              <Box ref={catalogScrollRef} sx={{ mt: 1, maxHeight: 460, overflowY: 'auto' }}>
                <Stack spacing={0.75}>
                {(catalogSource === 'suggested' ? moodboardSuggestions : catalogResults).map(result => {
                  const isSuggested = catalogSource === 'suggested'
                  const previews = isSuggested ? ((result as MoodboardSuggestion).preview_image_urls ?? []) : moodboardPreviews(result as MoodboardItem)
                  const description = isSuggested
                    ? ((result as MoodboardSuggestion).reason ?? '')
                    : ((result as MoodboardItem).taste_profile || (result as MoodboardItem).qwen_guidance?.prompt_guidance || (result as MoodboardItem).keywords.join(', '))
                  return (
                    <Box
                      key={result.id}
                      sx={{
                        border: '1px solid rgba(202,196,208,0.18)',
                        borderRadius: 1.5,
                        p: 1,
                        bgcolor: selectedCatalogIds.includes(result.id) ? 'rgba(187,134,252,0.12)' : 'rgba(255,255,255,0.03)',
                      }}
                    >
                      <Stack direction="row" spacing={1}>
                        {previews.length > 0 && (
                          <Box sx={{
                            flexShrink: 0, width: 62, aspectRatio: '2 / 3', borderRadius: 1, overflow: 'hidden',
                            display: 'grid', gridTemplateColumns: previews.length > 1 ? '1fr 1fr' : '1fr',
                            gap: '1px', bgcolor: 'background.default',
                          }}>
                            {previews.map((src, i) => (
                              <Box key={i} component="img" src={moodboardImageSrc(src)} alt="" loading="lazy"
                                sx={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                                onError={(e: any) => { e.target.style.visibility = 'hidden' }} />
                            ))}
                          </Box>
                        )}
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Stack direction="row" justifyContent="space-between" alignItems="flex-start" spacing={0.5}>
                            <Typography variant="body2" sx={{ fontWeight: 700 }}>{result.title}</Typography>
                            <Stack direction="row" spacing={0.25} alignItems="center" sx={{ flexShrink: 0 }}>
                              {!isSuggested && <Tooltip title={(result as MoodboardItem).favorite ? 'Remove from favorites' : 'Save to favorites'}>
                                <IconButton size="small" onClick={() => toggleCatalogFavorite(result as MoodboardItem)}>
                                  {(result as MoodboardItem).favorite
                                    ? <StarIcon fontSize="small" sx={{ color: '#f5c518' }} />
                                    : <StarBorderIcon fontSize="small" sx={{ color: 'text.disabled' }} />}
                                </IconButton>
                              </Tooltip>}
                              <Button
                                size="small"
                                variant={selectedCatalogIds.includes(result.id) ? 'contained' : 'outlined'}
                                disabled={catalogLoading}
                                onClick={() => selectedCatalogIds.includes(result.id)
                                  ? removeCatalogMoodboard(result.id)
                                  : isSuggested ? addSuggestedMoodboard(result as MoodboardSuggestion) : addCatalogMoodboard(result as MoodboardItem)}
                              >
                                {selectedCatalogIds.includes(result.id) ? 'Added' : 'Add'}
                              </Button>
                            </Stack>
                          </Stack>
                          {description && (
                            <Typography variant="caption" sx={{ color: 'text.secondary', display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical', overflow: 'hidden', mt: 0.25 }}>
                              {description}
                            </Typography>
                          )}
                          {!isSuggested && (result as MoodboardItem).keywords.length > 0 && (
                            <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
                              {(result as MoodboardItem).keywords.slice(0, 6).map(k => (
                                <Chip key={k} label={k} size="small" variant="outlined" sx={{ height: 18, fontSize: 10 }} />
                              ))}
                            </Stack>
                          )}
                        </Box>
                      </Stack>
                    </Box>
                  )
                })}
                </Stack>
                {catalogSource !== 'suggested' && catalogTotal > 0 && (
                  <Stack
                    ref={catalogSentinelRef}
                    direction="row"
                    alignItems="center"
                    justifyContent="space-between"
                    sx={{ mt: 1, pb: 0.5, pt: 0.5 }}
                  >
                    <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                      Showing {catalogResults.length} of {catalogTotal}
                      {catalogResults.length < catalogTotal ? ' · scroll for more' : ''}
                    </Typography>
                    {catalogResults.length < catalogTotal && (
                      <Button
                        size="small"
                        onClick={() => void searchCatalog(catalogQuery, catalogSource, catalogPage + 1, true)}
                        disabled={catalogLoading}
                      >
                        {catalogLoading ? <CircularProgress size={14} /> : 'Load more'}
                      </Button>
                    )}
                  </Stack>
                )}
              </Box>
            )}
          </Box>

          {/* Image board */}
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1 }}>
            Optional visual reference images
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.75 }}>
            Use these only when catalog moodboard text guidance is not strong enough. They are sent as direct Qwen visual references.
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap alignItems="center">
            {board.map((b64, i) => (
              <Box key={i} sx={{ position: 'relative', width: 56, height: 56, borderRadius: 1, overflow: 'hidden' }}>
                <img src={`data:image/png;base64,${b64}`} alt={`ref ${i + 1}`}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                <IconButton size="small" onClick={() => removeImage(i)}
                  sx={{ position: 'absolute', top: -2, right: -2, p: '1px', bgcolor: 'rgba(0,0,0,0.6)' }}>
                  <CloseIcon sx={{ fontSize: 12 }} />
                </IconButton>
              </Box>
            ))}
            <Tooltip title="Upload local images as direct visual style references for this generation.">
              <Button
                variant="outlined"
                startIcon={<AddPhotoAlternateIcon fontSize="small" />}
                onClick={() => fileRef.current?.click()}
                sx={{ minHeight: 56, borderStyle: 'dashed' }}
              >
                Add visual reference
              </Button>
            </Tooltip>
            <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={addImages} />
          </Stack>

          {/* Strength */}
          {active && (
            <Box sx={{ mt: 1.5 }}>
              <Stack direction="row" justifyContent="space-between">
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>Moodboard strength</Typography>
                <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12 }}>
                  {params.moodboard_strength.toFixed(2)}
                </Typography>
              </Stack>
              <Slider
                value={params.moodboard_strength} min={0} max={1} step={0.05}
                onChange={(_, v) => setParam('moodboard_strength', v as number)}
                size="small"
              />
              <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                0.35 = balanced default · higher = stronger style push
              </Typography>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  )
}
