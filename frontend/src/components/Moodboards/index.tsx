import { useCallback, useRef, useEffect, useState } from 'react'
import type { ChangeEvent } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardMedia,
  Chip,
  CircularProgress,
  Grid,
  IconButton,
  InputAdornment,
  Slider,
  Stack,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import FavoriteIcon from '@mui/icons-material/Favorite'
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder'
import RefreshIcon from '@mui/icons-material/Refresh'
import SearchIcon from '@mui/icons-material/Search'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import AddLinkIcon from '@mui/icons-material/AddLink'
import DeleteIcon from '@mui/icons-material/Delete'
import SaveAltIcon from '@mui/icons-material/SaveAlt'
import { apiFetch, type AuthSession, type MoodboardItem } from '../../api'
import { useStore } from '../../store'

const PAGE_SIZE = 48
const MAX_CUSTOM_MOODBOARD_REFS = 10

function previewImages(board: MoodboardItem): string[] {
  const images = board.image_urls.length ? board.image_urls : [board.primary_image_url].filter(Boolean)
  return images.slice(0, 4)
}

function moodboardErrorMessage(error: any, fallback: string) {
  const detail = error?.response?.data?.detail
  if (detail === 'Authentication required') {
    return 'Sign in to use shared moodboard actions, or run local mode for unauthenticated access.'
  }
  if (detail === 'Admin access required') {
    return 'Admin login is required to sync or import Krea moodboards in sharing mode.'
  }
  return detail ?? error?.message ?? fallback
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`))
    reader.onload = event => resolve(String(event.target?.result || '').split(',')[1] || '')
    reader.readAsDataURL(file)
  })
}

export default function MoodboardsPanel() {
  const [items, setItems] = useState<MoodboardItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [message, setMessage] = useState<{ severity: 'success' | 'error' | 'info'; text: string } | null>(null)
  const [loadError, setLoadError] = useState('')
  const [auth, setAuth] = useState<AuthSession | null>(null)
  const [mashupIds, setMashupIds] = useState<number[]>([])
  const [mashupWeights, setMashupWeights] = useState<Record<number, number>>({})
  const customFileRef = useRef<HTMLInputElement>(null)
  const { params, setParams, setTab, moodboardView, setMoodboardView } = useStore()
  const isAdmin = auth?.role === 'admin'

  const load = useCallback(async (pg = 1) => {
    setLoading(true)
    setLoadError('')
    try {
      if (moodboardView === 'new') {
        const discovery = await apiFetch.latestMoodboardDiscovery()
        const needle = query.trim().toLowerCase()
        const newItems = needle
          ? discovery.items.filter(board => [
            board.title,
            board.taste_profile,
            board.keywords.join(' '),
          ].join(' ').toLowerCase().includes(needle))
          : discovery.items
        setItems(newItems)
        setTotal(newItems.length)
        setPage(1)
        return
      }
      const data = await apiFetch.moodboards({
        q: query,
        page: pg,
        pageSize: PAGE_SIZE,
        favorites: moodboardView === 'favorites',
        source: moodboardView === 'custom' ? 'custom' : moodboardView === 'official' ? 'official' : undefined,
      })
      setItems(prev => pg === 1 ? data.items : [...prev, ...data.items])
      setTotal(data.total)
      setPage(pg)
    } catch (e: any) {
      const detail = moodboardErrorMessage(e, 'Could not load moodboards')
      setLoadError(detail)
      setMessage({ severity: 'error', text: detail })
    } finally {
      setLoading(false)
    }
  }, [moodboardView, query])

  useEffect(() => { load(1) }, [load])
  useEffect(() => {
    apiFetch.authMe().then(setAuth).catch(() => setAuth(null))
  }, [])

  const toggleFavorite = async (board: MoodboardItem) => {
    await apiFetch.setMoodboardFavorite(board.id, !board.favorite)
    setItems(prev => prev.map(item => item.id === board.id ? { ...item, favorite: !item.favorite } : item))
  }

  const refreshCatalog = async () => {
    setBusy('Syncing Krea moodboards')
    setMessage(null)
    try {
      const result = await apiFetch.importMoodboards()
      setMessage({
        severity: 'success',
        text: result.new_count > 0
          ? `Imported or updated ${result.imported} moodboards. ${result.new_count} are new.`
          : `Imported or updated ${result.imported} moodboards. No new moodboards found.`,
      })
      if (result.new_count > 0) setMoodboardView('new')
      await load(1)
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not sync moodboards') })
    } finally {
      setBusy(null)
    }
  }

  const importUrls = async () => {
    const text = window.prompt('Paste one or more Krea moodboard URLs, separated by commas or new lines.')
    if (!text?.trim()) return
    const urls = text.split(/[\n,]+/).map(url => url.trim()).filter(Boolean)
    setBusy('Importing moodboards')
    setMessage(null)
    try {
      const result = await apiFetch.importMoodboards(urls)
      setMessage({
        severity: 'success',
        text: result.new_count > 0
          ? `Imported or updated ${result.imported} moodboards. ${result.new_count} are new.`
          : `Imported or updated ${result.imported} moodboards. No new moodboards found.`,
      })
      if (result.new_count > 0) setMoodboardView('new')
      await load(1)
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not import moodboards') })
    } finally {
      setBusy(null)
    }
  }

  const exportSeed = async () => {
    setBusy('Exporting portable seed')
    setMessage(null)
    try {
      const result = await apiFetch.exportMoodboardSeed()
      setMessage({ severity: 'success', text: `Exported ${result.exported} moodboards to ${result.path}.` })
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not export moodboard seed') })
    } finally {
      setBusy(null)
    }
  }

  const createCustomMoodboard = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []).slice(0, MAX_CUSTOM_MOODBOARD_REFS)
    event.target.value = ''
    if (!files.length) return
    const title = window.prompt('Name this custom moodboard. Leave blank to let local Qwen name it.', '') ?? ''
    const tasteProfile = window.prompt('Optional taste profile / style description.', '') ?? ''
    const keywordsText = window.prompt('Optional keywords, separated by commas.', '') ?? ''
    setBusy('Saving custom moodboard')
    setMessage(null)
    try {
      const image_b64s = (await Promise.all(files.map(fileToBase64))).filter(Boolean)
      const created = await apiFetch.createCustomMoodboard({
        title: title.trim(),
        taste_profile: tasteProfile.trim(),
        keywords: keywordsText.split(',').map(keyword => keyword.trim()).filter(Boolean),
        image_b64s,
      })
      setMoodboardView('custom')
      setQuery('')
      setItems([created])
      setTotal(1)
      setPage(1)
      setMessage({ severity: 'success', text: `Saved custom moodboard “${created.title}”.` })
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not save custom moodboard') })
    } finally {
      setBusy(null)
    }
  }

  const deleteCustomMoodboard = async (board: MoodboardItem) => {
    if (board.source !== 'custom') return
    if (!window.confirm(`Delete custom moodboard “${board.title}”?`)) return
    setBusy(`Deleting ${board.title}`)
    setMessage(null)
    try {
      await apiFetch.deleteCustomMoodboard(board.id)
      setItems(prev => prev.filter(item => item.id !== board.id))
      setMashupIds(prev => prev.filter(id => id !== board.id))
      setTotal(prev => Math.max(0, prev - 1))
      setMessage({ severity: 'success', text: `Deleted custom moodboard “${board.title}”.` })
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not delete custom moodboard') })
    } finally {
      setBusy(null)
    }
  }

  const useMoodboard = async (board: MoodboardItem) => {
    setBusy(`Loading ${board.title}`)
    setMessage(null)
    try {
      const basePrompt = params.prompt.trim()
      setParams({
        mode: 'txt2img',
        mood: '',
        selected_moodboard_ids: [board.id],
        moodboard_uuids: board.uuid ? [board.uuid] : [],
        moodboard_strength: 0.35,
        moodboard_images: [],
        prompt: basePrompt || board.title,
      })
      setMessage({ severity: 'success', text: `Loaded ${board.title} style guidance.` })
      setTab(0)
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not load moodboard') })
    } finally {
      setBusy(null)
    }
  }

  const generateGuidance = async (board: MoodboardItem) => {
    setBusy(`Generating Qwen guidance for ${board.title}`)
    setMessage(null)
    try {
      const updated = await apiFetch.generateMoodboardGuidance(board.id)
      setItems(prev => prev.map(item => item.id === updated.id ? updated : item))
      setMessage({ severity: 'success', text: `Generated Qwen prompt guidance for “${updated.title}”.` })
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not generate Qwen guidance') })
    } finally {
      setBusy(null)
    }
  }

  const generateMissingGuidance = async () => {
    setBusy('Generating missing Qwen guidance')
    setMessage(null)
    try {
      const result = await apiFetch.generateMissingMoodboardGuidance(10)
      setMessage({ severity: 'success', text: `Generated Qwen guidance for ${result.processed} moodboards.` })
      await load(1)
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not generate missing Qwen guidance') })
    } finally {
      setBusy(null)
    }
  }

  const toggleMashup = (board: MoodboardItem) => {
    setMashupIds(prev => prev.includes(board.id)
      ? prev.filter(id => id !== board.id)
      : [...prev, board.id].slice(0, 5))
    setMashupWeights(prev => {
      const next = { ...prev }
      if (prev[board.id] != null) delete next[board.id]
      else next[board.id] = 1.0
      return next
    })
  }
  const mashupTitle = (id: number) => items.find(b => b.id === id)?.title || `Moodboard #${id}`

  const createMashup = async () => {
    if (mashupIds.length < 2) {
      setMessage({ severity: 'error', text: 'Select at least two moodboards to mash up.' })
      return
    }
    setBusy('Creating Qwen moodboard mashup')
    setMessage(null)
    try {
      const created = await apiFetch.createMoodboardMashup({
        moodboard_ids: mashupIds,
        weights: mashupIds.map(id => mashupWeights[id] ?? 1.0),
      })
      setMashupIds([])
      setMashupWeights({})
      setMoodboardView('custom')
      setQuery('')
      setItems([created])
      setTotal(1)
      setPage(1)
      setMessage({ severity: 'success', text: `Saved mashup moodboard “${created.title}”.` })
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not create moodboard mashup') })
    } finally {
      setBusy(null)
    }
  }

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 } }}>
      <Stack spacing={2}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5} justifyContent="space-between" alignItems={{ xs: 'stretch', md: 'center' }}>
          <Box>
            <Typography variant="h5">Moodboards</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Search public Krea moodboards, save favorites, and load their enriched style guidance into local txt2img conditioning.
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            <Button variant="outlined" onClick={createMashup} disabled={!!busy || mashupIds.length < 2}>
              Create mashup{mashupIds.length ? ` (${mashupIds.length})` : ''}
            </Button>
            <Button variant="contained" onClick={() => customFileRef.current?.click()} disabled={!!busy}>
              Create custom
            </Button>
            <input ref={customFileRef} type="file" accept="image/*" multiple hidden onChange={createCustomMoodboard} />
            {isAdmin && (
              <>
                <Button variant="outlined" onClick={generateMissingGuidance} disabled={!!busy}>
                  Generate missing guidance
                </Button>
                <Button variant="outlined" startIcon={<AddLinkIcon />} onClick={importUrls} disabled={!!busy}>
                  Import URLs
                </Button>
                <Button variant="outlined" startIcon={<SaveAltIcon />} onClick={exportSeed} disabled={!!busy}>
                  Export seed
                </Button>
                <Button variant="contained" startIcon={busy ? <CircularProgress size={16} /> : <RefreshIcon />} onClick={refreshCatalog} disabled={!!busy}>
                  Sync Krea
                </Button>
              </>
            )}
          </Stack>
        </Stack>

        {mashupIds.length >= 2 && (
          <Box sx={{ p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Mashup weights<Typography component="span" variant="caption" sx={{ color: 'text.secondary', ml: 1 }}>
                — how strongly each source influences the Qwen-synthesized blend
              </Typography>
            </Typography>
            <Stack spacing={1}>
              {mashupIds.map(id => (
                <Stack key={id} direction="row" spacing={2} alignItems="center">
                  <Typography variant="body2" sx={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {mashupTitle(id)}
                  </Typography>
                  <Slider
                    sx={{ width: 160 }}
                    size="small"
                    value={mashupWeights[id] ?? 1.0}
                    min={0.1} max={3.0} step={0.1}
                    valueLabelDisplay="auto"
                    onChange={(_, v) => setMashupWeights(prev => ({ ...prev, [id]: v as number }))}
                  />
                  <Typography variant="caption" sx={{ width: 28, fontFamily: 'Roboto Mono' }}>
                    {(mashupWeights[id] ?? 1.0).toFixed(1)}
                  </Typography>
                </Stack>
              ))}
            </Stack>
          </Box>
        )}

        {message && <Alert severity={message.severity} onClose={() => setMessage(null)}>{message.text}</Alert>}
        {busy && <Alert severity="info" icon={<CircularProgress size={18} />}>{busy}</Alert>}

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} alignItems={{ xs: 'stretch', sm: 'center' }}>
          <TextField
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') load(1) }}
            placeholder="Search moods, lighting, color, composition..."
            fullWidth
            InputProps={{
              startAdornment: <InputAdornment position="start"><SearchIcon /></InputAdornment>,
            }}
          />
          <Tabs value={moodboardView} onChange={(_, v) => setMoodboardView(v)} variant="scrollable" scrollButtons="auto" sx={{ minWidth: { xs: 0, sm: 360 }, maxWidth: '100%' }}>
            <Tab label="Official" value="official" />
            <Tab label="Favorites" value="favorites" />
            <Tab label="Custom" value="custom" />
            <Tab label="New" value="new" />
          </Tabs>
          <Tooltip title="Refresh list">
            <IconButton onClick={() => load(1)} disabled={loading}><RefreshIcon /></IconButton>
          </Tooltip>
        </Stack>

        {loading && page === 1 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
            <CircularProgress />
          </Box>
        ) : loadError ? (
          <Box sx={{ textAlign: 'center', py: 8, maxWidth: 720, mx: 'auto' }}>
            <Alert severity="error" sx={{ textAlign: 'left', mb: 2 }}>
              Moodboard catalog failed to load: {loadError}
            </Alert>
            <Typography sx={{ color: 'text.secondary', mb: 2 }}>
              This does not mean your local catalog was deleted. The official Krea moodboards are stored in the local database and can be retried without syncing.
            </Typography>
            <Stack direction="row" spacing={1} justifyContent="center">
              <Button variant="contained" onClick={() => load(1)}>Retry local catalog</Button>
              {isAdmin && <Button variant="outlined" onClick={refreshCatalog}>Sync Krea</Button>}
            </Stack>
          </Box>
        ) : items.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography sx={{ color: 'text.secondary' }}>
              {moodboardView === 'new'
                ? 'No new moodboards have been discovered yet.'
                : moodboardView === 'custom'
                  ? 'No custom moodboards yet. Click Create custom and choose reference images.'
                  : query.trim()
                    ? 'No moodboards matched that search. Clear the search or try a broader style word.'
                    : 'No moodboards are visible in this view. Try Retry local catalog; admins can use Sync Krea only to refresh from Krea.'}
            </Typography>
          </Box>
        ) : (
          <Grid container spacing={1.5}>
            {items.map(board => {
              const images = previewImages(board)
              return (
                <Grid item xs={12} sm={6} md={4} lg={3} key={board.id}>
                  <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', aspectRatio: '1.55 / 1', bgcolor: 'background.default' }}>
                      {images.length ? images.map((src, index) => (
                        <CardMedia
                          key={`${board.id}-${src}`}
                          component="img"
                          image={src}
                          alt={`${board.title} reference ${index + 1}`}
                          sx={{ height: '100%', minHeight: 0, objectFit: 'cover' }}
                        />
                      )) : (
                        <Box sx={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Typography variant="caption" sx={{ color: 'text.disabled' }}>No preview</Typography>
                        </Box>
                      )}
                    </Box>
                    <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Stack direction="row" justifyContent="space-between" spacing={1} alignItems="flex-start">
                        <Typography variant="h6" sx={{ fontSize: 16, lineHeight: 1.25 }}>{board.title}</Typography>
                        <Stack direction="row" spacing={0.25}>
                          {board.source === 'custom' && (
                            <IconButton size="small" onClick={() => deleteCustomMoodboard(board)} aria-label="Delete custom moodboard">
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          )}
                          <IconButton size="small" onClick={() => toggleFavorite(board)} aria-label={board.favorite ? 'Remove favorite' : 'Add favorite'}>
                            {board.favorite ? <FavoriteIcon fontSize="small" sx={{ color: '#F48FB1' }} /> : <FavoriteBorderIcon fontSize="small" />}
                          </IconButton>
                        </Stack>
                      </Stack>
                      <Typography variant="body2" sx={{ color: 'text.secondary', display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                        {board.taste_profile || 'No taste profile imported yet.'}
                      </Typography>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                        <Chip
                          label={board.qwen_guidance_version > 0 ? 'Qwen guidance' : 'Needs Qwen guidance'}
                          size="small"
                          color={board.qwen_guidance_version > 0 ? 'success' : 'warning'}
                          variant={board.qwen_guidance_version > 0 ? 'filled' : 'outlined'}
                        />
                        {mashupIds.includes(board.id) && <Chip label="Mashup source" size="small" color="primary" />}
                        {board.keywords.slice(0, 6).map(keyword => (
                          <Chip key={keyword} label={keyword} size="small" variant="outlined" />
                        ))}
                      </Stack>
                      <Box sx={{ flex: 1 }} />
                      {board.qwen_guidance?.prompt_guidance && (
                        <Typography variant="caption" sx={{ color: 'text.disabled', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                          {board.qwen_guidance.prompt_guidance}
                        </Typography>
                      )}
                      <Stack direction="row" spacing={1}>
                        <Button size="small" variant={mashupIds.includes(board.id) ? 'contained' : 'outlined'} onClick={() => toggleMashup(board)} disabled={!!busy}>
                          Mashup
                        </Button>
                        <Button size="small" variant="outlined" onClick={() => generateGuidance(board)} disabled={!!busy}>
                          Qwen guidance
                        </Button>
                      </Stack>
                      <Button startIcon={<AutoAwesomeIcon />} variant="contained" onClick={() => useMoodboard(board)} disabled={!!busy}>
                        Use moodboard
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              )
            })}
          </Grid>
        )}

        {items.length < total && (
          <Box sx={{ textAlign: 'center' }}>
            <Button variant="outlined" onClick={() => load(page + 1)} disabled={loading}>
              {loading ? <CircularProgress size={16} /> : 'Load more'}
            </Button>
          </Box>
        )}
      </Stack>
    </Box>
  )
}
