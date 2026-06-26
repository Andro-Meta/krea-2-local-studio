import React, { useEffect, useRef } from 'react'
import { Box, Button, Slider, Stack, Typography } from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import { useStore } from '../../store'
import MaskCanvas from './MaskCanvas'

export default function InpaintPanel() {
  const { params, setParam, setParams } = useStore()

  useEffect(() => { setParams({ mode: 'inpaint' }) }, [])

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => setParam('init_image_b64', (ev.target?.result as string).split(',')[1])
    reader.readAsDataURL(file)
  }

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 900, mx: 'auto' }}>
      <Stack spacing={2}>
        {!params.init_image_b64 ? (
          <Box
            sx={{ border: '2px dashed rgba(202,196,208,0.3)', borderRadius: 2, p: 3, textAlign: 'center', cursor: 'pointer', '&:hover': { borderColor: 'primary.main' } }}
            onClick={() => document.getElementById('inpaint-upload')?.click()}
          >
            <input id="inpaint-upload" type="file" accept="image/*" hidden onChange={handleFile} />
            <Stack alignItems="center" spacing={0.5}>
              <UploadFileIcon sx={{ color: 'text.secondary', fontSize: 40 }} />
              <Typography sx={{ color: 'text.secondary' }}>Upload image to inpaint</Typography>
            </Stack>
          </Box>
        ) : (
          <MaskCanvas
            imageB64={params.init_image_b64}
            onMaskChange={mask => setParam('mask_b64', mask)}
          />
        )}
        {params.init_image_b64 && (
          <Button variant="outlined" size="small" onClick={() => { setParam('init_image_b64', ''); setParam('mask_b64', '') }}>
            Clear image
          </Button>
        )}
      </Stack>
    </Box>
  )
}
