import React, { useEffect, useState } from 'react'
import {
  Box, Chip, CircularProgress, Collapse, IconButton, InputAdornment,
  Slider, Stack, TextField, Tooltip, Typography,
} from '@mui/material'
import DownloadIcon from '@mui/icons-material/Download'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import AddLinkIcon from '@mui/icons-material/AddLink'
import { useStore } from '../../store'
import { apiFetch } from '../../api'

export default function LoraSection() {
  const { params, setParam, loras, setLoras } = useStore()
  const [downloading, setDownloading] = useState<Record<string, boolean>>({})
  const [importUrl, setImportUrl] = useState('')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState('')
  const [showImport, setShowImport] = useState(false)

  const refresh = () => apiFetch.loras().then(setLoras).catch(() => {})

  useEffect(() => { refresh() }, [])

  const activeLora = (name: string) => params.loras.find(l => l.name === name)

  const toggleLora = (name: string, filename: string) => {
    const existing = params.loras.find(l => l.name === name)
    if (existing) {
      setParam('loras', params.loras.filter(l => l.name !== name))
    } else {
      setParam('loras', [...params.loras, { name, filename, strength: 1.0, enabled: true }])
    }
  }

  const setStrength = (name: string, strength: number) => {
    setParam('loras', params.loras.map(l => l.name === name ? { ...l, strength } : l))
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

  return (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block', textTransform: 'uppercase', letterSpacing: 1 }}>
        LoRAs
      </Typography>
      <Typography variant="caption" sx={{ color: 'text.disabled', mb: 1.5, display: 'block' }}>
        Style adapters. Click to toggle. Trigger words added to prompt automatically.
      </Typography>
      <Stack spacing={0.75}>
        {loras.map(lora => {
          const active = activeLora(lora.name)
          const isDownloading = downloading[lora.name]
          return (
            <Box key={lora.name}>
              <Stack direction="row" alignItems="center" spacing={1}>
                {lora.installed ? (
                  <Tooltip
                    title={lora.compatible === false
                      ? (lora.match_info ?? 'Not a Krea-2 LoRA — will not affect output')
                      : (lora.trigger_words.length
                        ? `Trigger word: "${lora.trigger_words.join(', ')}" — added to prompt automatically`
                        : lora.display_name)}
                    placement="right"
                    arrow
                  >
                    <Chip
                      label={lora.compatible === false ? `${lora.display_name} ⚠` : lora.display_name}
                      size="small"
                      variant={active ? 'filled' : 'outlined'}
                      color={lora.compatible === false ? 'warning' : (active ? 'secondary' : 'default')}
                      onClick={() => toggleLora(lora.name, lora.filename)}
                      clickable
                      icon={active ? <CheckCircleIcon sx={{ fontSize: '14px !important' }} /> : undefined}
                    />
                  </Tooltip>
                ) : (
                  <Tooltip title={`Not downloaded. Click to download from HuggingFace.`} placement="right" arrow>
                    <span>
                      <Chip
                        label={lora.display_name}
                        size="small"
                        variant="outlined"
                        color="default"
                        disabled
                        sx={{ opacity: 0.5 }}
                      />
                    </span>
                  </Tooltip>
                )}
                {!lora.installed && (
                  <Tooltip title={`Download ${lora.display_name}`} arrow>
                    <span>
                      <IconButton
                        size="small"
                        onClick={() => download(lora.name)}
                        disabled={isDownloading}
                        sx={{ p: 0.25 }}
                      >
                        {isDownloading
                          ? <CircularProgress size={14} />
                          : <DownloadIcon sx={{ fontSize: 16, color: 'text.disabled' }} />}
                      </IconButton>
                    </span>
                  </Tooltip>
                )}
              </Stack>
              {active && lora.installed && (
                <Stack direction="row" spacing={1} alignItems="center" sx={{ pl: 1, pt: 0.25, maxWidth: 200 }}>
                  <Typography variant="caption" sx={{ color: 'text.disabled', minWidth: 40 }}>
                    {active.strength.toFixed(2)}
                  </Typography>
                  <Slider
                    value={active.strength}
                    min={0} max={2} step={0.05}
                    onChange={(_, v) => setStrength(lora.name, v as number)}
                    size="small"
                    valueLabelDisplay="auto"
                  />
                </Stack>
              )}
            </Box>
          )
        })}
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
