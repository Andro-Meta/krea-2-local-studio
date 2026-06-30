import { Alert, Box, Button, CircularProgress, Snackbar, Tabs, Tab } from '@mui/material'
import { lazy, Suspense, useEffect, useState } from 'react'
import Layout from './components/Layout'
import GeneratePanel from './components/GeneratePanel'
import SystemStatus from './components/SystemStatus'
import RedrawStudio from './components/RedrawStudio'
import RealtimeStudio from './components/RealtimeStudio'
import Lightbox from './components/Gallery/Lightbox'
import { useStore } from './store'
import { apiFetch, type MoodboardDiscovery } from './api'

const SEEN_MOODBOARD_DISCOVERY_KEY = 'krea2_seen_moodboard_discovery_id'
const GalleryPanel = lazy(() => import('./components/Gallery'))
const MoodboardsPanel = lazy(() => import('./components/Moodboards'))

function LazyTabFallback() {
  return (
    <Box sx={{ display: 'grid', placeItems: 'center', minHeight: 280 }}>
      <CircularProgress size={28} />
    </Box>
  )
}

export default function App() {
  const { tab, lightbox, params, setParam, setParams, setTab, setMoodboardView, createMode, setCreateMode } = useStore()
  const [moodboardToast, setMoodboardToast] = useState<MoodboardDiscovery | null>(null)

  useEffect(() => {
    if (params.mode !== 'txt2img' && createMode !== 'realtime') setCreateMode('redraw')
  }, [createMode, params.mode])

  useEffect(() => {
    let cancelled = false
    apiFetch.settings()
      .then(settings => {
        if (cancelled) return
        const engine = settings.diffusion_engine ?? 'native_pytorch'
        setParams({
          diffusion_engine: engine,
          quantization: engine === 'native_int8_convrot' ? 'int8' : params.quantization === 'int8' ? 'fp8' : params.quantization,
        })
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let stopped = false
    const checkDiscovery = async () => {
      try {
        const latest = await apiFetch.latestMoodboardDiscovery()
        if (stopped || !latest.id || latest.new_count <= 0) return
        if (localStorage.getItem(SEEN_MOODBOARD_DISCOVERY_KEY) === latest.id) return
        localStorage.setItem(SEEN_MOODBOARD_DISCOVERY_KEY, latest.id)
        setMoodboardToast(latest)
      } catch {
        // Best-effort notification only; the Moodboards tab can still be opened normally.
      }
    }
    checkDiscovery()
    const interval = window.setInterval(checkDiscovery, 60_000)
    return () => {
      stopped = true
      window.clearInterval(interval)
    }
  }, [])

  const renderCreate = () => (
    <Box>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', px: { xs: 1.5, sm: 2 } }}>
        <Tabs
          value={createMode}
          onChange={(_, v) => {
            setCreateMode(v)
            if (v === 'txt2img') setParams({
              mode: 'txt2img',
              init_image_b64: '',
              mask_b64: '',
              bboxes: [],
              style_fusion_mode: 'semantic_fusion',
              edit_provider: 'auto',
              inpaint_method: 'native',
              differential_inpaint: false,
              moodboard_images: [],
            })
            if (v === 'redraw') setParam('mode', 'redraw')
          }}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Text → Image" value="txt2img" />
          <Tab label="Redraw Studio" value="redraw" />
          <Tab label="Realtime Studio" value="realtime" />
        </Tabs>
      </Box>
      {createMode === 'txt2img' && <GeneratePanel />}
      {createMode === 'redraw' && (
        <>
          {params.diffusion_engine === 'gguf_external' || params.diffusion_engine === 'int8_convrot_external' ? (
            <Box sx={{ p: 2 }}>
              <Alert
                severity="warning"
                action={<Button color="inherit" size="small" onClick={() => setParam('diffusion_engine', 'native_pytorch')}>Use Native</Button>}
              >
                Redraw, img2img, inpaint, and outpaint require a native Krea engine. GGUF/external engines are txt2img-only until benchmarks pass.
              </Alert>
            </Box>
          ) : (
            <>
              <RedrawStudio />
              <GeneratePanel />
            </>
          )}
        </>
      )}
      {createMode === 'realtime' && (
        params.diffusion_engine === 'gguf_external' || params.diffusion_engine === 'int8_convrot_external' ? (
          <Box sx={{ p: 2 }}>
            <Alert
              severity="warning"
              action={<Button color="inherit" size="small" onClick={() => setParam('diffusion_engine', 'native_pytorch')}>Use Native</Button>}
            >
              Realtime Studio currently requires native Krea Turbo. GGUF realtime stays disabled until low-VRAM benchmarks pass.
            </Alert>
          </Box>
        ) : <RealtimeStudio />
      )}
    </Box>
  )

  return (
    <Layout>
      {tab === 0 && renderCreate()}
      {tab === 1 && (
        <Suspense fallback={<LazyTabFallback />}>
          <GalleryPanel />
        </Suspense>
      )}
      {tab === 2 && (
        <Suspense fallback={<LazyTabFallback />}>
          <MoodboardsPanel />
        </Suspense>
      )}
      {tab === 3 && <SystemStatus />}
      {lightbox && <Lightbox />}
      <Snackbar open={!!moodboardToast} autoHideDuration={8000} onClose={() => setMoodboardToast(null)}>
        <Alert
          severity="success"
          variant="filled"
          onClose={() => setMoodboardToast(null)}
          action={
            <Button
              color="inherit"
              size="small"
              onClick={() => {
                setMoodboardView('new')
                setTab(2)
                setMoodboardToast(null)
              }}
            >
              View
            </Button>
          }
        >
          Krea found {moodboardToast?.new_count ?? 0} new moodboard{moodboardToast?.new_count === 1 ? '' : 's'}.
        </Alert>
      </Snackbar>
    </Layout>
  )
}
