import { Box, Tabs, Tab } from '@mui/material'
import { useEffect, useState } from 'react'
import Layout from './components/Layout'
import GeneratePanel from './components/GeneratePanel'
import GalleryPanel from './components/Gallery'
import MoodboardsPanel from './components/Moodboards'
import SystemStatus from './components/SystemStatus'
import RedrawStudio from './components/RedrawStudio'
import RealtimeStudio from './components/RealtimeStudio'
import Lightbox from './components/Gallery/Lightbox'
import { useStore } from './store'

export default function App() {
  const { tab, lightbox, params, setParam } = useStore()
  const [createMode, setCreateMode] = useState<'txt2img' | 'redraw' | 'realtime'>(
    params.mode === 'txt2img' ? 'txt2img' : 'redraw',
  )

  useEffect(() => {
    if (params.mode !== 'txt2img' && createMode !== 'realtime') setCreateMode('redraw')
  }, [createMode, params.mode])

  const renderCreate = () => (
    <Box>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', px: { xs: 1.5, sm: 2 } }}>
        <Tabs
          value={createMode}
          onChange={(_, v) => {
            setCreateMode(v)
            if (v === 'txt2img') setParam('mode', 'txt2img')
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
          <RedrawStudio />
          <GeneratePanel />
        </>
      )}
      {createMode === 'realtime' && <RealtimeStudio />}
    </Box>
  )

  return (
    <Layout>
      {tab === 0 && renderCreate()}
      {tab === 1 && <GalleryPanel />}
      {tab === 2 && <MoodboardsPanel />}
      {tab === 3 && <SystemStatus />}
      {lightbox && <Lightbox />}
    </Layout>
  )
}
