import React, { useEffect, useMemo, useState } from 'react'
import {
  Box, Button, Chip, CircularProgress, Collapse, Dialog, DialogContent, DialogTitle,
  IconButton, InputAdornment, LinearProgress, MenuItem, Slider, Stack, TextField,
  ToggleButton, ToggleButtonGroup, Tooltip, Typography,
} from '@mui/material'
import DownloadIcon from '@mui/icons-material/Download'
import CheckIcon from '@mui/icons-material/Check'
import AddLinkIcon from '@mui/icons-material/AddLink'
import SyncIcon from '@mui/icons-material/Sync'
import TravelExploreIcon from '@mui/icons-material/TravelExplore'
import OpenInNewIcon from '@mui/icons-material/OpenInNew'
import CloseIcon from '@mui/icons-material/Close'
import SearchIcon from '@mui/icons-material/Search'
import VpnKeyIcon from '@mui/icons-material/VpnKey'
import { useStore } from '../../store'
import { apiFetch, type CivitaiLoraItem, type LoraInfo } from '../../api'

const BLOCK_FILTERS = ['all', 'style_safe', 'early', 'middle', 'late'] as const

// A gradient placeholder for LoRAs with no Civitai preview.
function loraGradient(seed: string): string {
  let h = 0
  for (const c of seed) h = (h * 31 + c.charCodeAt(0)) % 360
  return `linear-gradient(135deg, hsl(${h},45%,32%), hsl(${(h + 40) % 360},50%,20%))`
}

// ---------------------------------------------------------------------------
// Civitai browse + install dialog (Krea 2 LoRA / LoKr only)
// ---------------------------------------------------------------------------

function CivitaiBrowseDialog({ open, onClose, onInstalled }: {
  open: boolean; onClose: () => void; onInstalled: () => void
}) {
  const [q, setQ] = useState('')
  const [sort, setSort] = useState('Most Downloaded')
  const [items, setItems] = useState<CivitaiLoraItem[]>([])
  const [loading, setLoading] = useState(false)
  const [installing, setInstalling] = useState<number | null>(null)
  const [installed, setInstalled] = useState<Record<number, boolean>>({})
  const [error, setError] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [hasKey, setHasKey] = useState(false)
  const [savingKey, setSavingKey] = useState(false)
  const [showKey, setShowKey] = useState(false)

  const search = async () => {
    setLoading(true); setError('')
    try { setItems((await apiFetch.civitaiLoras({ query: q.trim(), sort })).items) }
    catch { setError('Civitai search failed. Check your connection.') }
    setLoading(false)
  }
  useEffect(() => { if (open) search() }, [open, sort])
  useEffect(() => {
    if (!open) return
    apiFetch.settings().then(s => setHasKey(!!s.has_civitai_token)).catch(() => {})
  }, [open])

  const saveKey = async () => {
    setSavingKey(true); setError('')
    try {
      await apiFetch.updateSettings({ civitai_token: apiKey.trim() })
      setHasKey(!!apiKey.trim())
      setApiKey(''); setShowKey(false)
      await search()
    } catch { setError('Could not save the Civitai API key.') }
    setSavingKey(false)
  }

  const install = async (it: CivitaiLoraItem) => {
    setInstalling(it.version_id); setError('')
    try {
      await apiFetch.civitaiInstall(it.version_id, it.file_name || undefined)
      setInstalled(m => ({ ...m, [it.version_id]: true }))
      onInstalled()
    } catch (e: any) {
      const status = e?.response?.status
      if (status === 401 || status === 403) {
        setShowKey(true)
        setError('This model requires a (free) Civitai API key. Paste one below and Save, then try again.')
      } else {
        setError(e?.response?.data?.detail ?? 'Install failed. Check your connection.')
      }
    }
    setInstalling(null)
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth PaperProps={{ sx: { borderRadius: 3, bgcolor: '#17181d' } }}>
      <DialogTitle sx={{ pb: 1 }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Stack direction="row" spacing={1} alignItems="center">
            <TravelExploreIcon sx={{ color: 'primary.main' }} />
            <span>Browse Krea 2 LoRAs</span>
          </Stack>
          <IconButton size="small" onClick={onClose}><CloseIcon fontSize="small" /></IconButton>
        </Stack>
      </DialogTitle>
      <DialogContent>
        <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
          <TextField
            size="small" fullWidth autoFocus placeholder="Search Krea 2 LoRA / LoKr models…"
            value={q} onChange={e => setQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && search()}
            InputProps={{ startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" /></InputAdornment> }}
          />
          <TextField select size="small" value={sort} onChange={e => setSort(e.target.value)} sx={{ minWidth: 160 }}>
            {['Most Downloaded', 'Highest Rated', 'Newest'].map(s => <MenuItem key={s} value={s}>{s}</MenuItem>)}
          </TextField>
        </Stack>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.disabled' }}>
            Base model “Krea 2” · LoRA + LoKr. Some downloads need a free Civitai API key.
          </Typography>
          <Button size="small" startIcon={<VpnKeyIcon sx={{ fontSize: 15 }} />} onClick={() => setShowKey(v => !v)}
            sx={{ textTransform: 'none', color: hasKey ? 'success.main' : 'text.secondary', flexShrink: 0 }}>
            {hasKey ? 'API key saved' : 'Add API key'}
          </Button>
        </Stack>
        <Collapse in={showKey}>
          <Stack direction="row" spacing={1} sx={{ mb: 1 }} alignItems="flex-start">
            <TextField
              size="small" fullWidth type="password" value={apiKey}
              onChange={e => setApiKey(e.target.value)} onKeyDown={e => e.key === 'Enter' && saveKey()}
              placeholder={hasKey ? 'Key available. Paste a new key to replace it for this session.' : 'Paste your Civitai API key...'}
              helperText={<>Free: create one at <a href="https://civitai.com/user/account" target="_blank" rel="noreferrer" style={{ color: '#9ecbff' }}>civitai.com - Account - API Keys</a>. Session keys apply immediately; persistent CIVITAI_TOKEN values in .env are preserved.</>}
              InputProps={{ startAdornment: <InputAdornment position="start"><VpnKeyIcon fontSize="small" /></InputAdornment> }}
            />
            <Button variant="contained" size="small" onClick={saveKey} disabled={savingKey} sx={{ mt: 0.25 }}
              startIcon={savingKey ? <CircularProgress size={12} color="inherit" /> : undefined}>
              Save
            </Button>
          </Stack>
        </Collapse>
        {error && <Typography variant="caption" sx={{ color: 'warning.main', mb: 1, display: 'block' }}>{error}</Typography>}
        {loading ? (
          <Stack alignItems="center" sx={{ py: 5 }}><CircularProgress /></Stack>
        ) : (
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr 1fr', sm: 'repeat(3, 1fr)' }, gap: 1.25 }}>
            {items.map(it => {
              const isInstalled = !!(it.installed || installed[it.version_id])
              return (
                <Box key={it.version_id} sx={{ borderRadius: 2, overflow: 'hidden', border: '1px solid', borderColor: 'divider', bgcolor: 'rgba(255,255,255,0.02)' }}>
                  <Box sx={{ position: 'relative', pt: '100%', background: loraGradient(it.name) }}>
                    {it.preview_url && (
                      <Box component="img" src={it.preview_url} alt=""
                        sx={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
                        onError={(e: any) => { e.target.style.display = 'none' }} />
                    )}
                    <Box sx={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0) 55%)' }} />
                    {it.nsfw && <Chip label="NSFW" size="small" color="error" sx={{ position: 'absolute', top: 6, left: 6, height: 18, fontSize: 10 }} />}
                    <Typography variant="caption" sx={{ position: 'absolute', bottom: 6, left: 8, right: 8, color: '#fff', fontWeight: 700, lineHeight: 1.15, textShadow: '0 1px 3px rgba(0,0,0,0.8)' }}>
                      {it.name}
                    </Typography>
                  </Box>
                  <Box sx={{ p: 1 }}>
                    <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block' }} noWrap>
                      {it.creator || 'unknown'} · {(it.downloads ?? 0).toLocaleString()} downloads
                    </Typography>
                    {it.trigger_words?.length > 0 && (
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.25 }} noWrap title={it.trigger_words.join(', ')}>
                        Trigger: {it.trigger_words.join(', ')}
                      </Typography>
                    )}
                    {isInstalled && it.installed_filename && (
                      <Typography variant="caption" sx={{ color: 'success.main', display: 'block', mt: 0.25 }} noWrap title={it.installed_filename}>
                        Installed as {it.installed_filename}
                      </Typography>
                    )}
                    <Stack direction="row" spacing={0.5} sx={{ mt: 0.75 }} alignItems="center">
                      <Button fullWidth size="small" variant="contained"
                        disabled={installing === it.version_id || isInstalled}
                        onClick={() => install(it)}
                        startIcon={installing === it.version_id ? <CircularProgress size={12} /> : (isInstalled ? <CheckIcon sx={{ fontSize: 15 }} /> : <DownloadIcon sx={{ fontSize: 15 }} />)}>
                        {isInstalled ? 'Installed' : 'Install'}
                      </Button>
                      <Tooltip title="Open on Civitai" arrow>
                        <IconButton size="small" component="a" href={it.civitai_url} target="_blank" rel="noreferrer"><OpenInNewIcon sx={{ fontSize: 15 }} /></IconButton>
                      </Tooltip>
                    </Stack>
                  </Box>
                </Box>
              )
            })}
            {!items.length && <Typography variant="caption" sx={{ color: 'text.disabled', gridColumn: '1 / -1' }}>No Krea 2 LoRAs matched.</Typography>}
          </Box>
        )}
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------

export default function LoraSection() {
  const { params, setParam, loras, setLoras } = useStore()
  const [downloading, setDownloading] = useState<Record<string, boolean>>({})
  const [importUrl, setImportUrl] = useState('')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState('')
  const [showImport, setShowImport] = useState(false)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<'all' | 'attached' | 'krea2'>('all')
  const [syncing, setSyncing] = useState(false)
  const [syncMsg, setSyncMsg] = useState('')
  const [browseOpen, setBrowseOpen] = useState(false)

  const refresh = () => apiFetch.loras().then(setLoras).catch(() => {})
  useEffect(() => { refresh() }, [])

  const attachedFor = (name: string) => params.loras.find(l => l.name === name)
  const infoFor = (name: string) => loras.find(l => l.name === name)

  const syncCivitai = async () => {
    setSyncing(true); setSyncMsg('Hashing & matching on Civitai…')
    try {
      await apiFetch.lorasCivitaiScan()
      for (let i = 0; i < 120; i++) {
        await new Promise(r => setTimeout(r, 2000))
        const s = await apiFetch.lorasCivitaiScanStatus()
        setSyncMsg(`Matching ${s.done}/${s.total} · ${s.updated} identified`)
        await refresh()
        if (!s.scanning) break
      }
    } catch { setSyncMsg('Sync failed') }
    setSyncing(false); setTimeout(() => setSyncMsg(''), 2500)
  }

  const toggleLora = (lora: LoraInfo) => {
    if (attachedFor(lora.name)) setParam('loras', params.loras.filter(l => l.name !== lora.name))
    else setParam('loras', [...params.loras, { name: lora.name, filename: lora.filename, strength: lora.strength ?? 1.0, enabled: true, block_filter: 'style_safe' }])
  }
  const setStrength = (name: string, strength: number) => setParam('loras', params.loras.map(l => l.name === name ? { ...l, strength } : l))
  const setBlockFilter = (name: string, bf: typeof BLOCK_FILTERS[number]) => setParam('loras', params.loras.map(l => l.name === name ? { ...l, block_filter: bf } : l))

  const download = async (name: string) => {
    setDownloading(d => ({ ...d, [name]: true }))
    try { await apiFetch.downloadLora(name); await refresh() }
    catch (e: any) { alert(`Download failed: ${e?.response?.data?.detail ?? e.message}`) }
    setDownloading(d => ({ ...d, [name]: false }))
  }

  const handleImport = async () => {
    if (!importUrl.trim()) return
    setImporting(true); setImportError('')
    try {
      const r = await apiFetch.importLoraUrl(importUrl.trim())
      await refresh()
      if (r.compatible === false) setImportError(`Downloaded, but ${r.match_info ?? 'not a Krea-2 LoRA'}`)
      else { setImportUrl(''); setShowImport(false) }
    } catch (e: any) { setImportError(e?.response?.data?.detail ?? e.message ?? 'Import failed') }
    setImporting(false)
  }

  const needle = query.trim().toLowerCase()
  const matches = (lora: LoraInfo) => {
    if (needle && ![lora.display_name, lora.name, lora.filename, lora.base_model, ...(lora.trigger_words ?? []), lora.description ?? '', lora.match_info ?? '']
      .join(' ').toLowerCase().includes(needle)) return false
    if (filter === 'attached') return !!attachedFor(lora.name)
    if (filter === 'krea2') return lora.base_model === 'Krea 2' || lora.is_official
    return true
  }
  const installedLoras = useMemo(() => loras.filter(l => l.installed && matches(l)), [loras, needle, filter, params.loras])
  const missingLoras = useMemo(() => loras.filter(l => !l.installed && matches(l)), [loras, needle, filter])

  if (!loras.length) return null

  // ---- attached editor row ------------------------------------------------
  const renderAttachedEditor = (attached: NonNullable<ReturnType<typeof attachedFor>>) => {
    const info = infoFor(attached.name)
    const isBypass = attached.name === 'krea2filterbypass3'
    const min = isBypass ? -40000 : -4, max = isBypass ? 40000 : 4, step = isBypass ? 1 : 0.05
    return (
      <Box key={attached.name} sx={{ p: 1, borderRadius: 2, bgcolor: 'rgba(187,134,252,0.08)', border: '1px solid', borderColor: 'primary.main' }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <Box sx={{ width: 34, height: 34, borderRadius: 1, flexShrink: 0, background: loraGradient(attached.name), backgroundSize: 'cover',
            ...(info?.preview_url ? { backgroundImage: `url(${info.preview_url})`, backgroundPosition: 'center' } : {}) }} />
          <Typography variant="body2" sx={{ fontWeight: 600, flex: 1, minWidth: 0 }} noWrap>{info?.display_name ?? attached.name}</Typography>
          <Chip label={isBypass ? Math.round(attached.strength).toLocaleString() : attached.strength.toFixed(2)}
            size="small" color="secondary" sx={{ height: 20, fontVariantNumeric: 'tabular-nums' }} />
          <IconButton size="small" onClick={() => toggleLora(info ?? { name: attached.name, filename: attached.filename } as LoraInfo)}><CloseIcon sx={{ fontSize: 15 }} /></IconButton>
        </Stack>
        <Box sx={{ px: 1, mt: 0.5 }}>
          <Slider value={attached.strength} min={min} max={max} step={step}
            marks={isBypass ? [{ value: -40000, label: '-40k' }, { value: 0, label: '0' }, { value: 40000, label: '+40k' }]
              : [{ value: -2, label: 'avoid' }, { value: 0, label: 'off' }, { value: 2, label: 'apply' }, { value: 4, label: 'max' }]}
            onChange={(_, v) => setStrength(attached.name, v as number)} size="small" valueLabelDisplay="auto"
            sx={{ '& .MuiSlider-markLabel': { fontSize: 10 } }} />
        </Box>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mt: 0.25 }}>
          {isBypass && (
            <TextField size="small" type="number" label="Exact strength" value={attached.strength}
              onChange={e => setStrength(attached.name, Math.max(-40000, Math.min(40000, Number(e.target.value) || 0)))}
              inputProps={{ min: -40000, max: 40000, step: 1 }} sx={{ flex: 1 }} />
          )}
          <TextField select size="small" label="Block filter" value={attached.block_filter ?? 'all'}
            onChange={e => setBlockFilter(attached.name, e.target.value as typeof BLOCK_FILTERS[number])} sx={{ flex: 1 }}>
            {BLOCK_FILTERS.map(f => <MenuItem key={f} value={f}>{f.replace('_', '-')}</MenuItem>)}
          </TextField>
        </Stack>
      </Box>
    )
  }

  // ---- installed grid card ------------------------------------------------
  const renderCard = (lora: LoraInfo) => {
    const attached = !!attachedFor(lora.name)
    const incompatible = lora.compatible === false
    const tip = incompatible ? (lora.match_info ?? 'Not a Krea-2 LoRA') :
      [lora.description, lora.trigger_words.length ? `Trigger: ${lora.trigger_words.join(', ')}` : ''].filter(Boolean).join('\n\n') || lora.display_name
    return (
      <Tooltip key={lora.name} title={<span style={{ whiteSpace: 'pre-line' }}>{tip}</span>} placement="top" arrow enterDelay={400}>
        <Box
          onClick={() => !incompatible && toggleLora(lora)}
          sx={{
            position: 'relative', borderRadius: 2, overflow: 'hidden', cursor: incompatible ? 'not-allowed' : 'pointer',
            border: '2px solid', borderColor: attached ? 'primary.main' : 'transparent',
            outline: '1px solid', outlineColor: 'divider', opacity: incompatible ? 0.5 : 1,
            transition: 'transform .12s, box-shadow .12s',
            '&:hover': incompatible ? {} : { transform: 'translateY(-2px)', boxShadow: '0 6px 18px rgba(0,0,0,0.45)' },
          }}
        >
          <Box sx={{ position: 'relative', pt: '78%', background: loraGradient(lora.name) }}>
            {lora.preview_url && (
              <Box component="img" src={lora.preview_url} alt=""
                sx={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
                onError={(e: any) => { e.target.style.display = 'none' }} />
            )}
            <Box sx={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0.15) 100%)' }} />
            {/* top row: base badge + attached check */}
            {lora.base_model && (
              <Chip label={lora.base_model} size="small"
                color={lora.base_model === 'Krea 2' ? 'success' : 'default'}
                sx={{ position: 'absolute', top: 6, left: 6, height: 18, fontSize: 10, fontWeight: 700 }} />
            )}
            {lora.civitai_url && (
              <IconButton size="small" component="a" href={lora.civitai_url} target="_blank" rel="noreferrer"
                onClick={e => e.stopPropagation()}
                sx={{ position: 'absolute', top: 2, right: 2, color: 'rgba(255,255,255,0.75)', '&:hover': { color: '#fff' } }}>
                <OpenInNewIcon sx={{ fontSize: 14 }} />
              </IconButton>
            )}
            {attached && (
              <Box sx={{ position: 'absolute', bottom: 6, right: 6, width: 22, height: 22, borderRadius: '50%', bgcolor: 'primary.main', display: 'grid', placeItems: 'center' }}>
                <CheckIcon sx={{ fontSize: 15, color: '#000' }} />
              </Box>
            )}
            <Typography variant="caption" sx={{ position: 'absolute', bottom: 5, left: 8, right: 32, color: '#fff', fontWeight: 700, lineHeight: 1.15, textShadow: '0 1px 3px rgba(0,0,0,0.9)' }}>
              {incompatible ? `${lora.display_name} ⚠` : lora.display_name}
            </Typography>
          </Box>
          {lora.trigger_words.length > 0 && (
            <Box sx={{ px: 0.75, py: 0.5, bgcolor: 'rgba(255,255,255,0.03)' }}>
              <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: 10.5 }} noWrap title={lora.trigger_words.join(', ')}>
                🔑 {lora.trigger_words.join(', ')}
              </Typography>
            </Box>
          )}
        </Box>
      </Tooltip>
    )
  }

  const attachedList = params.loras
  return (
    <Box>
      {/* Toolbar */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1, fontWeight: 700 }}>
          LoRA Library{attachedList.length ? ` · ${attachedList.length} attached` : ''}
        </Typography>
        <Stack direction="row" spacing={0.5}>
          <Tooltip title="Match installed LoRAs to Civitai (names, triggers, previews)" arrow>
            <Button size="small" startIcon={syncing ? <CircularProgress size={12} /> : <SyncIcon sx={{ fontSize: 16 }} />} onClick={syncCivitai} disabled={syncing}>Sync</Button>
          </Tooltip>
          <Button size="small" variant="contained" startIcon={<TravelExploreIcon sx={{ fontSize: 16 }} />} onClick={() => setBrowseOpen(true)}>Browse Civitai</Button>
        </Stack>
      </Stack>
      {syncing && <LinearProgress sx={{ mb: 0.75, borderRadius: 1 }} />}
      {syncMsg && <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.75 }}>{syncMsg}</Typography>}

      {/* Search + filters */}
      <Stack direction="row" spacing={1} sx={{ mb: 1 }} alignItems="center">
        <TextField size="small" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search LoRAs, triggers, descriptions…" fullWidth
          InputProps={{ startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" /></InputAdornment> }} />
        <ToggleButtonGroup size="small" exclusive value={filter} onChange={(_, v) => v && setFilter(v)}>
          <ToggleButton value="all" sx={{ px: 1.25, textTransform: 'none' }}>All</ToggleButton>
          <ToggleButton value="attached" sx={{ px: 1.25, textTransform: 'none' }}>Attached</ToggleButton>
          <ToggleButton value="krea2" sx={{ px: 1.25, textTransform: 'none' }}>Krea 2</ToggleButton>
        </ToggleButtonGroup>
      </Stack>

      {/* Attached editors */}
      {attachedList.length > 0 && (
        <Stack spacing={0.75} sx={{ mb: 1.25 }}>
          {attachedList.map(renderAttachedEditor)}
        </Stack>
      )}

      {/* Installed grid */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(128px, 1fr))', gap: 1 }}>
        {installedLoras.map(renderCard)}
      </Box>
      {!installedLoras.length && (
        <Typography variant="caption" sx={{ color: 'text.disabled' }}>
          {needle || filter !== 'all' ? 'No LoRAs match this filter.' : 'No LoRAs installed yet — use Browse Civitai.'}
        </Typography>
      )}

      {/* Optional downloads */}
      {missingLoras.length > 0 && (
        <Box sx={{ mt: 1.25 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, display: 'block', mb: 0.5 }}>
            Available downloads ({missingLoras.length})
          </Typography>
          <Stack spacing={0.5}>
            {missingLoras.map(lora => (
              <Stack key={lora.name} direction="row" alignItems="center" justifyContent="space-between"
                sx={{ p: 0.75, borderRadius: 1.5, border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="caption" sx={{ minWidth: 0 }} noWrap>{lora.display_name}</Typography>
                {lora.download_enabled !== false && (
                  <IconButton size="small" onClick={() => download(lora.name)} disabled={downloading[lora.name]}>
                    {downloading[lora.name] ? <CircularProgress size={14} /> : <DownloadIcon sx={{ fontSize: 16 }} />}
                  </IconButton>
                )}
              </Stack>
            ))}
          </Stack>
        </Box>
      )}

      {/* URL import */}
      <Stack direction="row" alignItems="center" spacing={0.5} sx={{ mt: 1 }}>
        <Tooltip title="Import from HuggingFace or CivitAI URL" arrow placement="right">
          <IconButton size="small" onClick={() => { setShowImport(v => !v); setImportError('') }} sx={{ p: 0.25 }}>
            <AddLinkIcon sx={{ fontSize: 16, color: showImport ? 'primary.main' : 'text.disabled' }} />
          </IconButton>
        </Tooltip>
        {!showImport && <Typography variant="caption" sx={{ color: 'text.disabled', cursor: 'pointer' }} onClick={() => setShowImport(true)}>Import from URL</Typography>}
      </Stack>
      <Collapse in={showImport}>
        <TextField
          size="small" placeholder="Paste HuggingFace or CivitAI URL…" value={importUrl} sx={{ mt: 0.75 }}
          onChange={e => { setImportUrl(e.target.value); setImportError('') }}
          onKeyDown={e => e.key === 'Enter' && handleImport()} fullWidth error={!!importError}
          helperText={importError || 'HF: .../blob/main/file.safetensors  ·  CivitAI: civitai.com/models/…?modelVersionId=…'}
          InputProps={{ endAdornment: (
            <InputAdornment position="end">
              <IconButton size="small" onClick={handleImport} disabled={importing || !importUrl.trim()}>
                {importing ? <CircularProgress size={14} /> : <DownloadIcon sx={{ fontSize: 16 }} />}
              </IconButton>
            </InputAdornment>
          ) }}
        />
      </Collapse>

      <CivitaiBrowseDialog open={browseOpen} onClose={() => setBrowseOpen(false)} onInstalled={refresh} />
    </Box>
  )
}
