import React, { useState } from 'react'
import { Alert, Box, CircularProgress, IconButton, Snackbar, Stack, TextField, Tooltip } from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import { useStore } from '../../store'
import { apiFetch } from '../../api'

export default function PromptSection() {
  const { params, setParam } = useStore()
  const [expanding, setExpanding] = useState(false)
  const [notice, setNotice] = useState<{ message: string; severity: 'success' | 'warning' | 'error' } | null>(null)

  const handleExpand = async () => {
    if (!params.prompt || expanding) return
    setExpanding(true)
    try {
      const { expanded, changed, error, backend } = await apiFetch.expandPrompt(params.prompt)
      if (changed && expanded) {
        setParam('prompt', expanded)
        const label = backend === 'openrouter' ? 'OpenRouter' : backend === 'ideogram-json' ? 'Ideogram JSON' : 'Local Qwen3-VL'
        setNotice({ severity: 'success', message: `Prompt expanded with ${label}.` })
      } else if (error) {
        setNotice({ severity: 'warning', message: error })
      } else {
        setNotice({ severity: 'warning', message: 'The wand did not return a different prompt.' })
      }
    } catch (err) {
      setNotice({ severity: 'error', message: err instanceof Error ? err.message : 'Prompt expansion failed.' })
    } finally {
      setExpanding(false)
    }
  }

  return (
    <Stack spacing={1}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
        <TextField
          label="Prompt"
          multiline
          minRows={3}
          maxRows={8}
          fullWidth
          value={params.prompt}
          onChange={e => setParam('prompt', e.target.value)}
          placeholder="Describe the image you want to create…"
        />
        <Tooltip title="Expand prompt with AI">
          <span>
            <IconButton onClick={handleExpand} disabled={expanding || !params.prompt} sx={{ mt: 0.5 }}>
              {expanding ? <CircularProgress size={20} /> : <AutoFixHighIcon />}
            </IconButton>
          </span>
        </Tooltip>
      </Box>
      {params.mode !== 'txt2img' && (
        <TextField
          label="Negative prompt"
          multiline
          minRows={1}
          maxRows={3}
          fullWidth
          value={params.negative_prompt}
          onChange={e => setParam('negative_prompt', e.target.value)}
          placeholder="What to avoid…"
          size="small"
        />
      )}
      <Snackbar
        open={!!notice}
        autoHideDuration={5000}
        onClose={() => setNotice(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        {notice ? (
          <Alert severity={notice.severity} variant="filled" onClose={() => setNotice(null)}>
            {notice.message}
          </Alert>
        ) : undefined}
      </Snackbar>
    </Stack>
  )
}
