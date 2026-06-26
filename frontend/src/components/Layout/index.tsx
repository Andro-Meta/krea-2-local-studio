import React from 'react'
import { BottomNavigation, BottomNavigationAction, Box, Drawer, List, ListItemButton, ListItemIcon, ListItemText, Toolbar, useMediaQuery, useTheme } from '@mui/material'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import CollectionsIcon from '@mui/icons-material/Collections'
import MonitorIcon from '@mui/icons-material/Monitor'
import { useStore } from '../../store'

const TABS = [
  { label: 'Create', icon: <AutoAwesomeIcon /> },
  { label: 'Gallery', icon: <CollectionsIcon /> },
  { label: 'System', icon: <MonitorIcon /> },
]

const RAIL_WIDTH = 72

export default function Layout({ children }: { children: React.ReactNode }) {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const { tab, setTab } = useStore()

  if (isMobile) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Box sx={{ flex: 1, pb: '56px', overflowY: 'auto' }}>{children}</Box>
        <BottomNavigation
          value={tab}
          onChange={(_, v) => setTab(v)}
          sx={{ position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 1200 }}
        >
          {TABS.map(t => (
            <BottomNavigationAction key={t.label} label={t.label} icon={t.icon} />
          ))}
        </BottomNavigation>
      </Box>
    )
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: RAIL_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: RAIL_WIDTH,
            backgroundColor: '#1E1B2E',
            borderRight: '1px solid rgba(202,196,208,0.12)',
            overflowX: 'hidden',
          },
        }}
      >
        <Toolbar sx={{ minHeight: '48px !important' }} />
        <List sx={{ pt: 1 }}>
          {TABS.map((t, i) => (
            <ListItemButton
              key={t.label}
              selected={tab === i}
              onClick={() => setTab(i)}
              sx={{
                flexDirection: 'column',
                py: 1.5,
                px: 0,
                borderRadius: 2,
                mx: 0.5,
                mb: 0.5,
                '&.Mui-selected': {
                  backgroundColor: 'rgba(208,188,255,0.16)',
                  '& .MuiListItemIcon-root': { color: '#D0BCFF' },
                  '& .MuiListItemText-primary': { color: '#D0BCFF', fontWeight: 500 },
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 0, justifyContent: 'center', color: 'text.secondary' }}>{t.icon}</ListItemIcon>
              <ListItemText
                primary={t.label}
                primaryTypographyProps={{ variant: 'caption', fontSize: 10, textAlign: 'center' }}
              />
            </ListItemButton>
          ))}
        </List>
      </Drawer>
      <Box component="main" sx={{ flex: 1, minWidth: 0, overflowY: 'auto' }}>{children}</Box>
    </Box>
  )
}
