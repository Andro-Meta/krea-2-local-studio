import React, { useEffect, useRef, useState } from 'react'
import {
  Alert, Box, Button, Chip, CircularProgress, Collapse, IconButton, Slider, Stack, TextField, Tooltip, Typography,
} from '@mui/material'
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate'
import CloseIcon from '@mui/icons-material/Close'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import { useStore } from '../../store'
import { apiFetch, type Mood, type MoodboardItem } from '../../api'

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
  intro = 'Search official Krea moodboards here, add one or more styles, then generate normally. Catalog boards use style guidance only by default, so they should not copy the source moodboard image layout.',
  promptValue,
  onPromptFallback,
  applyTitleToPrompt = true,
}: MoodboardSectionProps) {
  const { params, setParam } = useStore()
  const [moods, setMoods] = useState<Mood[]>([])
  const [catalogQuery, setCatalogQuery] = useState('')
  const [catalogResults, setCatalogResults] = useState<MoodboardItem[]>([])
  const [selectedBoards, setSelectedBoards] = useState<MoodboardItem[]>([])
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [mashupLoading, setMashupLoading] = useState(false)
  const [catalogMessage, setCatalogMessage] = useState('')
  const [open, setOpen] = useState(true)
  const fileRef = useRef<HTMLInputElement>(null)

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

  const searchCatalog = async (query = catalogQuery) => {
    setCatalogLoading(true)
    setCatalogMessage('')
    try {
      const data = await apiFetch.moodboards({ q: query, page: 1, pageSize: 12 })
      setCatalogResults(data.items)
      if (!data.items.length) setCatalogMessage('No catalog moodboards matched that search.')
    } catch (e: any) {
      setCatalogMessage(moodboardErrorMessage(e, 'Could not search moodboards.'))
    } finally {
      setCatalogLoading(false)
    }
  }

  useEffect(() => {
    if (!open || catalogResults.length) return
    searchCatalog('')
    // Only auto-load a small browse set when the section is first opened.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

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
          <Alert severity="info" sx={{ py: 0.75, mb: 1.5 }}>
            {intro}
          </Alert>

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

          {/* Mood presets, grouped by category */}
          <Box sx={{ maxHeight: 280, overflowY: 'auto', mb: 1.5 }}>
            {[...new Set(moods.map(m => m.category))].map(cat => (
              <Box key={cat} sx={{ mb: 1 }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, fontWeight: 600 }}>
                  {cat}
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {moods.filter(m => m.category === cat).map(m => (
                    <Tooltip key={m.id} arrow placement="top"
                      title={<><b>Adds:</b> {m.keywords}<br /><b>Avoids:</b> {m.avoids}</>}>
                      <Chip
                        label={`${m.emoji} ${m.name}`}
                        size="small"
                        clickable
                        variant={selectedIds.includes(m.id) ? 'filled' : 'outlined'}
                        color={selectedIds.includes(m.id) ? (m.category === 'Horror' ? 'error' : 'secondary') : 'default'}
                        onClick={() => pickMood(m.id)}
                      />
                    </Tooltip>
                  ))}
                </Box>
              </Box>
            ))}
          </Box>

          {/* Krea catalog moodboards */}
          <Box sx={{ mb: 1.5 }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontWeight: 600 }}>
                Search Official Krea Moodboards
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                Click Add to apply. Select 2+ to mash up.
              </Typography>
            </Stack>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={0.75}>
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
            </Stack>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>
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
            </Stack>
            {catalogMessage && (
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
                {catalogMessage}
              </Typography>
            )}
            {catalogResults.length > 0 && (
              <Stack spacing={0.75} sx={{ mt: 1, maxHeight: 260, overflowY: 'auto' }}>
                {catalogResults.map(result => (
                  <Box
                    key={result.id}
                    sx={{
                      border: '1px solid rgba(202,196,208,0.18)',
                      borderRadius: 1.5,
                      p: 1,
                      bgcolor: selectedCatalogIds.includes(result.id) ? 'rgba(187,134,252,0.12)' : 'rgba(255,255,255,0.03)',
                    }}
                  >
                    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} justifyContent="space-between" alignItems={{ xs: 'stretch', sm: 'center' }}>
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 700 }}>{result.title}</Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
                          {(result.qwen_guidance?.prompt_guidance || result.taste_profile || result.keywords.join(', ')).slice(0, 160)}
                        </Typography>
                      </Box>
                      <Button
                        size="small"
                        variant={selectedCatalogIds.includes(result.id) ? 'contained' : 'outlined'}
                        disabled={catalogLoading}
                        onClick={() => selectedCatalogIds.includes(result.id) ? removeCatalogMoodboard(result.id) : addCatalogMoodboard(result)}
                      >
                        {selectedCatalogIds.includes(result.id) ? 'Added' : 'Add'}
                      </Button>
                    </Stack>
                  </Box>
                ))}
              </Stack>
            )}
          </Box>

          {/* Image board */}
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
            <Tooltip title="Optional: upload reference images only when you want stronger visual pull than style text guidance.">
              <IconButton onClick={() => fileRef.current?.click()}
                sx={{ width: 56, height: 56, border: '1px dashed rgba(202,196,208,0.4)', borderRadius: 1 }}>
                <AddPhotoAlternateIcon fontSize="small" />
              </IconButton>
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
                0.35 = Comfy default · higher = stronger style push
              </Typography>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  )
}
