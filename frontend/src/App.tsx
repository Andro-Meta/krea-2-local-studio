import React from 'react'
import { Box, Tabs, Tab, Typography } from '@mui/material'
import Layout from './components/Layout'
import GeneratePanel from './components/GeneratePanel'
import GalleryPanel from './components/Gallery'
import SystemStatus from './components/SystemStatus'
import ImageToImagePanel from './components/ImageToImage'
import InpaintPanel from './components/Inpaint'
import OutpaintPanel from './components/Outpaint'
import Lightbox from './components/Gallery/Lightbox'
import { useStore } from './store'

export default function App() {
  const { tab, lightbox, params, setParam } = useStore()

  const renderCreate = () => (
    <Box>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', px: { xs: 1.5, sm: 2 } }}>
        <Tabs value={params.mode} onChange={(_, v) => setParam('mode', v)} variant="scrollable" scrollButtons="auto">
          <Tab label="Text → Image" value="txt2img" />
          <Tab label="Image → Image" value="img2img" />
          <Tab label="Inpaint" value="inpaint" />
          <Tab label="Outpaint" value="outpaint" />
        </Tabs>
      </Box>
      {params.mode === 'txt2img' && <GeneratePanel />}
      {params.mode === 'img2img' && (
        <>
          <ImageToImagePanel />
          <GeneratePanel />
        </>
      )}
      {params.mode === 'inpaint' && (
        <>
          <InpaintPanel />
          <GeneratePanel />
        </>
      )}
      {params.mode === 'outpaint' && (
        <>
          <OutpaintPanel />
          <GeneratePanel />
        </>
      )}
    </Box>
  )

  return (
    <Layout>
      {tab === 0 && renderCreate()}
      {tab === 1 && <GalleryPanel />}
      {tab === 2 && <SystemStatus />}
      {lightbox && <Lightbox />}
    </Layout>
  )
}
