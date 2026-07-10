import { useRef, useState, type ChangeEvent } from 'react'
import { Alert, Button, CircularProgress, Stack, TextField, Tooltip } from '@mui/material'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import { apiFetch } from '../api'

function readFileB64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('Could not read image'))
    reader.onload = ev => resolve(String(ev.target?.result ?? '').split(',')[1])
    reader.readAsDataURL(file)
  })
}

interface Props {
  value?: string
  onChange: (prompt: string) => void
  mode?: 'replace' | 'append'
  label?: string
  size?: 'small' | 'medium'
  compact?: boolean
  // When true, shows an optional guidance field: what to focus on or change.
  // Left blank, the model writes the full prompt from the image (as before).
  withGuidance?: boolean
}

export default function CreatePromptFromImage({
  value = '',
  onChange,
  mode = 'replace',
  label = 'Create prompt from image',
  size = 'small',
  compact = false,
  withGuidance = false,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [loading, setLoading] = useState(false)
  const [guidance, setGuidance] = useState('')
  const [error, setError] = useState<string | null>(null)

  const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    if (mode === 'replace' && value.trim() && !window.confirm('Replace the current prompt with one created from this image?')) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      const imageB64 = await readFileB64(file)
      const result = await apiFetch.describeImage(imageB64, 'recreate', withGuidance ? guidance.trim() : '')
      const next = mode === 'append' && value.trim()
        ? `${value.trim()}\n${result.prompt}`
        : result.prompt
      onChange(next)
    } catch (err: any) {
      // 409 = a generation is using the GPU; the helper waits its turn on purpose.
      const detail = err?.response?.data?.detail
      setError(err?.response?.status === 409
        ? (detail || 'Busy generating — try again once the current image finishes.')
        : (detail || err?.message || 'Could not read the image.'))
    } finally {
      setLoading(false)
    }
  }

  const button = (
    <Tooltip title={error || label} arrow>
      <span>
        <Button
          size={size}
          variant="outlined"
          startIcon={loading ? <CircularProgress size={14} /> : <AutoAwesomeIcon fontSize="small" />}
          onClick={() => inputRef.current?.click()}
          disabled={loading}
        >
          {compact ? 'Image prompt' : label}
        </Button>
      </span>
    </Tooltip>
  )

  if (!withGuidance) {
    return (
      <>
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={handleFile} />
        {button}
      </>
    )
  }

  return (
    <>
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={handleFile} />
      <Stack spacing={1} sx={{ minWidth: { sm: 260 } }}>
        <TextField
          size="small"
          fullWidth
          multiline
          maxRows={3}
          value={guidance}
          onChange={e => setGuidance(e.target.value)}
          disabled={loading}
          placeholder="Optional: what to focus on or change (leave blank for a full auto prompt)"
        />
        {button}
        {error && <Alert severity="info" sx={{ py: 0 }} onClose={() => setError(null)}>{error}</Alert>}
      </Stack>
    </>
  )
}
