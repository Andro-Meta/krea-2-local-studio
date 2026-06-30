import React, { useEffect, useState } from 'react'
import {
  Box, Chip, CircularProgress, Collapse, IconButton, InputAdornment,
  MenuItem, Slider, Stack, TextField, Tooltip, Typography,
} from '@mui/material'
import DownloadIcon from '@mui/icons-material/Download'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import AddLinkIcon from '@mui/icons-material/AddLink'
import { useStore } from '../../store'
import { apiFetch } from '../../api'

const BLOCK_FILTERS = ['all', 'style_safe', 'early', 'middle', 'late'] as const

export default function LoraSection() {
  const { params, setParam, loras, setLoras } = useStore()
  const [downloading, setDownloading] = useState<Record<string, boolean>>({})
  const [importUrl, setImportUrl] = useState('')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState('')
  const [showImport, setShowImport] = useState(false)
  const [query, setQuery] = useState('')

  const refresh = () => apiFetch.loras().then(setLoras).catch(() => {})

  useEffect(() => { refresh() }, [])

  const activeLora = (name: string) => params.loras.find(l => l.name === name)

  const toggleLora = (name: string, filename: string) => {
    const existing = params.loras.find(l => l.name === name)
    if (existing) {
      setParam('loras', params.loras.filter(l => l.name !== name))
    } else {
      const info = loras.find(lora => lora.name === name)
      setParam('loras', [...params.loras, { name, filename, strength: info?.strength ?? 1.0, enabled: true, block_filter: 'style_safe' }])
    }
  }

  const setStrength = (name: string, strength: number) => {
    setParam('loras', params.loras.map(l => l.name === name ? { ...l, strength } : l))
  }

  const setBlockFilter = (name: string, blockFilter: typeof BLOCK_FILTERS[number]) => {
    setParam('loras', params.loras.map(l => l.name === name ? { ...l, block_filter: blockFilter } : l))
  }

  const download = async (name: string) => {
    setDownloading(d => ({ ...d, [name]: true }))
    try {
      await apiFetch.downloadLora(name)
      await refresh()
    } catch (e: any) {
      alert(`Download failed: ${e?.response?.data?.detail ?? e.message}`)
    }
    setDownloading(d => ({ ...d, [name]: false }))
  }

  const handleImport = async () => {
    if (!importUrl.trim()) return
    setImporting(true)
    setImportError('')
    try {
      const r = await apiFetch.importLoraUrl(importUrl.trim())
      await refresh()
      if (r.compatible === false) {
        // Downloaded, but it isn't a real Krea-2 LoRA — keep panel open and warn.
        setImportError(`Downloaded, but ${r.match_info ?? 'not a Krea-2 LoRA'}`)
      } else {
        setImportUrl('')
        setShowImport(false)
      }
    } catch (e: any) {
      setImportError(e?.response?.data?.detail ?? e.message ?? 'Import failed')
    }
    setImporting(false)
  }

  if (!loras.length) return null

  const needle = query.trim().toLowerCase()
  const visibleLoras = loras.filter(lora => {
    if (!needle) return true
    return [
      lora.display_name,
      lora.name,
      lora.filename,
      lora.is_official ? 'official krea 2' : 'local',
      ...(lora.trigger_words ?? []),
      lora.match_info ?? '',
    ].join(' ').toLowerCase().includes(needle)
  })
  const installed = visibleLoras.filter(lora => lora.installed)
  const missing = visibleLoras.filter(lora => !lora.installed)

  const renderLora = (lora: typeof loras[number]) => {
    const active = activeLora(lora.name)
    const isDownloading = downloading[lora.name]
    const disabled = lora.installed && lora.compatible === false
    const canDownload = lora.download_enabled !== false
    return (
      <Box
        key={lora.name}
        sx={{
          p: 0.85,
          border: '1px solid',
          borderColor: active ? 'primary.main' : 'divider',
          borderRadius: 2,
          bgcolor: active ? 'rgba(187,134,252,0.10)' : 'rgba(255,255,255,0.02)',
        }}
      >
        <Stack direction="row" alignItems="center" spacing={1} justifyContent="space-between">
          <Box sx={{ minWidth: 0 }}>
            <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap>
              <Tooltip
                title={lora.compatible === false
                  ? (lora.match_info ?? 'Not a Krea-2 LoRA — will not affect output')
                  : (lora.trigger_words.length
                    ? `Trigger word: "${lora.trigger_words.join(', ')}" — added only for positive strength`
                    : lora.display_name)}
                placement="right"
                arrow
              >
                <span>
                  <Chip
                    label={lora.compatible === false ? `${lora.display_name} ⚠` : lora.display_name}
                    size="small"
                    variant={active ? 'filled' : 'outlined'}
                    color={lora.compatible === false ? 'warning' : (active ? 'secondary' : 'default')}
                    onClick={() => lora.installed && !disabled && toggleLora(lora.name, lora.filename)}
                    clickable={lora.installed && !disabled}
                    disabled={!lora.installed || disabled}
                    icon={active ? <CheckCircleIcon sx={{ fontSize: '14px !important' }} /> : undefined}
                  />
                </span>
              </Tooltip>
              <Chip
                label={lora.is_official ? 'Official Krea 2' : 'Local'}
                size="small"
                variant="outlined"
                sx={{ height: 20, fontSize: 11, opacity: 0.8 }}
              />
              {!lora.installed && <Chip label={canDownload ? 'Not downloaded' : 'Manual import required'} size="small" variant="outlined" sx={{ height: 20, fontSize: 11, opacity: 0.65 }} />}
            </Stack>
            {lora.trigger_words.length > 0 && (
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.35 }}>
                Trigger: {lora.trigger_words.join(', ')}
              </Typography>
            )}
            {lora.match_info && (!lora.is_official || !lora.installed) && (
              <Typography variant="caption" sx={{ color: lora.compatible === false ? 'warning.main' : 'text.disabled', display: 'block', mt: 0.35 }}>
                {lora.match_info}
              </Typography>
            )}
          </Box>
          {!lora.installed && canDownload && (
            <Tooltip title={`Download ${lora.display_name}`} arrow>
              <span>
                <IconButton
                  size="small"
                  onClick={() => download(lora.name)}
                  disabled={isDownloading}
                  sx={{ p: 0.5 }}
                >
                  {isDownloading
                    ? <CircularProgress size={16} />
                    : <DownloadIcon sx={{ fontSize: 18, color: 'text.disabled' }} />}
                </IconButton>
              </span>
            </Tooltip>
          )}
        </Stack>
        {active && lora.installed && (
          <Stack spacing={0.5} sx={{ pt: 0.85 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Strength: {active.strength.toFixed(2)}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                negative avoids · positive applies
              </Typography>
            </Stack>
            <Slider
              value={active.strength}
              min={-2} max={2} step={0.05}
              marks={[
                { value: -1, label: 'avoid' },
                { value: 0, label: 'off' },
                { value: 1, label: 'apply' },
              ]}
              onChange={(_, v) => setStrength(lora.name, v as number)}
              size="small"
              valueLabelDisplay="auto"
            />
            <TextField
              select
              size="small"
              label="Block filter"
              value={active.block_filter ?? 'all'}
              onChange={e => setBlockFilter(lora.name, e.target.value as typeof BLOCK_FILTERS[number])}
              helperText={(active.block_filter ?? 'all') === 'style_safe' ? 'Recommended for Krea style LoRAs' : undefined}
              sx={{ maxWidth: 220 }}
            >
              {BLOCK_FILTERS.map(filter => (
                <MenuItem key={filter} value={filter}>{filter.replace('_', '-')}</MenuItem>
              ))}
            </TextField>
          </Stack>
        )}
      </Box>
    )
  }

  const renderGroup = (title: string, items: typeof loras) => (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.75, fontWeight: 700 }}>
        {title} ({items.length})
      </Typography>
      <Stack spacing={0.75}>
        {items.length ? items.map(renderLora) : (
          <Typography variant="caption" sx={{ color: 'text.disabled' }}>
            {needle ? 'No LoRAs matched this search.' : 'None here yet.'}
          </Typography>
        )}
      </Stack>
    </Box>
  )

  return (
    <Box>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.75 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
          LoRA Library{params.loras.length ? ` · ${params.loras.length} attached` : ''}
        </Typography>
      </Stack>
      <Typography variant="caption" sx={{ color: 'text.disabled', mb: 1, display: 'block' }}>
        Attach multiple style adapters from `models/loras`. Positive strength applies a LoRA; negative strength pushes away from it.
      </Typography>
      <TextField
        size="small"
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Search installed, official, local, trigger words..."
        fullWidth
        sx={{ mb: 1 }}
      />
      <Stack spacing={1.25}>
        {renderGroup('Installed', installed)}
        {renderGroup('Available official downloads', missing)}
      </Stack>

      {/* URL import */}
      <Stack direction="row" alignItems="center" spacing={0.5} sx={{ mt: 1 }}>
        <Tooltip title="Import LoRA from HuggingFace or CivitAI URL" arrow placement="right">
          <IconButton size="small" onClick={() => { setShowImport(v => !v); setImportError('') }} sx={{ p: 0.25 }}>
            <AddLinkIcon sx={{ fontSize: 16, color: showImport ? 'primary.main' : 'text.disabled' }} />
          </IconButton>
        </Tooltip>
        {!showImport && (
          <Typography variant="caption" sx={{ color: 'text.disabled', cursor: 'pointer' }} onClick={() => setShowImport(true)}>
            Import from URL
          </Typography>
        )}
      </Stack>
      <Collapse in={showImport}>
        <Stack spacing={0.75} sx={{ mt: 0.75 }}>
          <TextField
            size="small"
            placeholder="Paste HuggingFace or CivitAI URL…"
            value={importUrl}
            onChange={e => { setImportUrl(e.target.value); setImportError('') }}
            onKeyDown={e => e.key === 'Enter' && handleImport()}
            fullWidth
            error={!!importError}
            helperText={importError || 'HF: .../blob/main/file.safetensors  ·  CivitAI: civitai.com/models/…?modelVersionId=…'}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton size="small" onClick={handleImport} disabled={importing || !importUrl.trim()}>
                    {importing ? <CircularProgress size={14} /> : <DownloadIcon sx={{ fontSize: 16 }} />}
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
        </Stack>
      </Collapse>
    </Box>
  )
}
