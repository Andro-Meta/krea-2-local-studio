import { useCallback, useEffect, type ChangeEvent } from 'react'
import { Box, Button, Slider, Stack, Typography } from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import { useStore } from '../../store'
import GeneratePanel from '../GeneratePanel'

function ImageUploader({ label, value, onChange }: { label: string; value: string; onChange: (b64: string) => void }) {
  const handleFile = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => {
      const result = ev.target?.result as string
      onChange(result.split(',')[1])
    }
    reader.readAsDataURL(file)
  }, [onChange])

  return (
    <Box
      sx={{
        border: '2px dashed rgba(202,196,208,0.3)', borderRadius: 2, p: 2,
        textAlign: 'center', cursor: 'pointer', position: 'relative',
        '&:hover': { borderColor: 'primary.main' },
      }}
      onClick={() => document.getElementById(`upload-${label}`)?.click()}
    >
      <input id={`upload-${label}`} type="file" accept="image/*" hidden onChange={handleFile} />
      {value ? (
        <img src={`data:image/png;base64,${value}`} alt={label} style={{ maxWidth: '100%', maxHeight: 200, borderRadius: 8 }} />
      ) : (
        <Stack alignItems="center" spacing={0.5}>
          <UploadFileIcon sx={{ color: 'text.secondary' }} />
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>{label}</Typography>
        </Stack>
      )}
    </Box>
  )
}

export default function ImageToImagePanel() {
  const { params, setParam, setParams } = useStore()

  useEffect(() => {
    setParams({ mode: 'img2img' })
  }, [])

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 900, mx: 'auto' }}>
      <Stack spacing={2}>
        <ImageUploader label="Upload image" value={params.init_image_b64} onChange={v => setParam('init_image_b64', v)} />
        <Box>
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2">Denoise strength</Typography>
            <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12 }}>{params.denoise}</Typography>
          </Stack>
          <Slider value={params.denoise} min={0.01} max={1.0} step={0.01}
            onChange={(_, v) => setParam('denoise', v as number)} size="small" />
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Lower = stays closer to original · Higher = more creative freedom
          </Typography>
        </Box>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Reference images (1–3) for style/content guidance:
        </Typography>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
          {([1, 2, 3] as const).map(n => {
            const key = `ref_image${n}_b64` as const
            return <Box key={n} sx={{ flex: 1 }}>
              <ImageUploader label={`Ref ${n}`} value={params[key]} onChange={v => setParam(key, v)} />
            </Box>
          })}
        </Stack>
      </Stack>
    </Box>
  )
}
