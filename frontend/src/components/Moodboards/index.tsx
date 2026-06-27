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
import { apiFetch, type MoodboardItem } from '../../api'
import { useStore } from '../../store'

const PAGE_SIZE = 48
const MAX_LOCAL_MOODBOARD_REFS = 10

function previewImages(board: MoodboardItem): string[] {
  const images = board.image_urls.length ? board.image_urls : [board.primary_image_url].filter(Boolean)
  return images.slice(0, 4)
}

function moodboardRefs(board: MoodboardItem): string[] {
  return Array.from(new Set([board.primary_image_url, ...board.image_urls].filter(Boolean))).slice(0, MAX_LOCAL_MOODBOARD_REFS)
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
  const customFileRef = useRef<HTMLInputElement>(null)
  const { params, setParams, setTab, moodboardView, setMoodboardView } = useStore()

  const load = useCallback(async (pg = 1) => {
    setLoading(true)
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
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not load moodboards') })
    } finally {
      setLoading(false)
    }
  }, [moodboardView, query])

  useEffect(() => { load(1) }, [load])

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
    const files = Array.from(event.target.files ?? []).slice(0, MAX_LOCAL_MOODBOARD_REFS)
    event.target.value = ''
    if (!files.length) return
    const title = window.prompt('Name this custom moodboard.')
    if (!title?.trim()) return
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
      setTotal(prev => Math.max(0, prev - 1))
      setMessage({ severity: 'success', text: `Deleted custom moodboard “${board.title}”.` })
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not delete custom moodboard') })
    } finally {
      setBusy(null)
    }
  }

  const useMoodboard = async (board: MoodboardItem) => {
    const refs = moodboardRefs(board)
    setBusy(`Loading ${board.title}`)
    setMessage(null)
    try {
      const images = await Promise.all(refs.map(src => apiFetch.moodboardImage(src)))
      const basePrompt = params.prompt.trim()
      setParams({
        mode: 'txt2img',
        mood: '',
        selected_moodboard_ids: [board.id],
        moodboard_uuids: board.uuid ? [board.uuid] : [],
        moodboard_strength: 0.35,
        moodboard_images: images,
        prompt: basePrompt || board.title,
      })
      setMessage({ severity: 'success', text: `Loaded ${images.length} local reference images from ${board.title}.` })
      setTab(0)
    } catch (e: any) {
      setMessage({ severity: 'error', text: moodboardErrorMessage(e, 'Could not load moodboard images') })
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
              Search public Krea moodboards, save favorites, and load up to {MAX_LOCAL_MOODBOARD_REFS} reference images into local txt2img Qwen conditioning.
            </Typography>
          </Box>
          <Stack direction="row" spacing={1}>
            <Button variant="contained" onClick={() => customFileRef.current?.click()} disabled={!!busy}>
              Create custom
            </Button>
            <input ref={customFileRef} type="file" accept="image/*" multiple hidden onChange={createCustomMoodboard} />
            <Button variant="outlined" startIcon={<AddLinkIcon />} onClick={importUrls} disabled={!!busy}>
              Import URLs
            </Button>
            <Button variant="outlined" startIcon={<SaveAltIcon />} onClick={exportSeed} disabled={!!busy}>
              Export seed
            </Button>
            <Button variant="contained" startIcon={busy ? <CircularProgress size={16} /> : <RefreshIcon />} onClick={refreshCatalog} disabled={!!busy}>
              Sync Krea
            </Button>
          </Stack>
        </Stack>

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
          <Tabs value={moodboardView} onChange={(_, v) => setMoodboardView(v)} sx={{ minWidth: 360 }}>
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
        ) : items.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography sx={{ color: 'text.secondary' }}>
              {moodboardView === 'new'
                ? 'No new moodboards have been discovered yet.'
                : moodboardView === 'custom'
                  ? 'No custom moodboards yet. Click Create custom and choose reference images.'
                  : 'No moodboards yet. Click Sync Krea or import a moodboard URL.'}
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
                        {board.keywords.slice(0, 6).map(keyword => (
                          <Chip key={keyword} label={keyword} size="small" variant="outlined" />
                        ))}
                      </Stack>
                      <Box sx={{ flex: 1 }} />
                      <Button startIcon={<AutoAwesomeIcon />} variant="contained" onClick={() => useMoodboard(board)} disabled={!!busy || !(board.image_urls.length || board.primary_image_url)}>
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
