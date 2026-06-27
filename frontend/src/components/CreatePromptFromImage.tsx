import { useRef, useState, type ChangeEvent } from 'react'
import { Button, CircularProgress, Tooltip } from '@mui/material'
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
}

export default function CreatePromptFromImage({
  value = '',
  onChange,
  mode = 'replace',
  label = 'Create prompt from image',
  size = 'small',
  compact = false,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [loading, setLoading] = useState(false)

  const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    if (mode === 'replace' && value.trim() && !window.confirm('Replace the current prompt with one created from this image?')) {
      return
    }
    setLoading(true)
    try {
      const imageB64 = await readFileB64(file)
      const result = await apiFetch.describeImage(imageB64)
      const next = mode === 'append' && value.trim()
        ? `${value.trim()}\n${result.prompt}`
        : result.prompt
      onChange(next)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={handleFile} />
      <Tooltip title={label}>
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
    </>
  )
}
