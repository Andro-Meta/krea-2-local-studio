import { Box, Tabs, Tab } from '@mui/material'
import Layout from './components/Layout'
import GeneratePanel from './components/GeneratePanel'
import GalleryPanel from './components/Gallery'
import SystemStatus from './components/SystemStatus'
import RedrawStudio from './components/RedrawStudio'
import Lightbox from './components/Gallery/Lightbox'
import { useStore } from './store'

export default function App() {
  const { tab, lightbox, params, setParam } = useStore()
  const createMode = params.mode === 'txt2img' ? 'txt2img' : 'redraw'

  const renderCreate = () => (
    <Box>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', px: { xs: 1.5, sm: 2 } }}>
        <Tabs
          value={createMode}
          onChange={(_, v) => setParam('mode', v)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Text → Image" value="txt2img" />
          <Tab label="Redraw Studio" value="redraw" />
        </Tabs>
      </Box>
      {params.mode === 'txt2img' && <GeneratePanel />}
      {params.mode !== 'txt2img' && (
        <>
          <RedrawStudio />
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
