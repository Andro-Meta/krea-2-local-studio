import type { GenerationRequest } from '../../api'

export type CharacterEditTask = NonNullable<GenerationRequest['character_edit_task']>

// identity = drop a person into a new scene (face matters)
// inplace  = edit an existing image in place (upload the image, not a face)
// compose  = place multiple people via boxes
export type CharacterEditKind = 'identity' | 'inplace' | 'compose'

export interface CharacterEditPreset {
  id: CharacterEditTask
  label: string
  prompt: string
  checkpoint: 'turbo' | 'raw'
  steps: number
  cfg: number
  groundingPx: number
  notes: string
  kind: CharacterEditKind
  sourceLabel: string
  sourceHint: string
}

export const characterEditPresets: CharacterEditPreset[] = [
  {
    id: 'restage',
    label: 'Restage person',
    prompt: 'Create a cinematic rainy night market portrait of the same adult person from the source image. Preserve face shape, hair, facial hair if present, eye color, eyebrows, nose, lips, skin texture, and calm direct expression. Change only the environment, lighting, outfit styling, and weather.',
    checkpoint: 'turbo',
    steps: 8,
    cfg: 1.0,
    groundingPx: 1536,
    notes: 'Best for new scenes + new clothes. Grounding 1536 + subject lock gives the strongest likeness transfer.',
    kind: 'identity',
    sourceLabel: 'Upload Person / Face',
    sourceHint: 'Upload or paste (Ctrl/Cmd+V) a clear photo of the person to restage',
  },
  {
    id: 'local_edit',
    label: 'Local edit',
    prompt: 'Add subtle round sunglasses to the person while preserving their face, hair, expression, pose, clothing, lighting, and the rest of the image exactly.',
    checkpoint: 'turbo',
    steps: 8,
    cfg: 1.0,
    groundingPx: 768,
    notes: 'Edit an existing photo in place: accessories, color changes, small object edits.',
    kind: 'inplace',
    sourceLabel: 'Upload Image to Edit',
    sourceHint: 'Upload or paste (Ctrl/Cmd+V) the image you want to edit',
  },
  {
    id: 'replace',
    label: 'Replace object',
    prompt: 'Replace the object in the person’s hand with a red rose. Preserve the same face, hair, facial hair if present, pose, lighting, clothing, and background.',
    checkpoint: 'turbo',
    steps: 8,
    cfg: 1.0,
    groundingPx: 768,
    notes: 'Edit an existing photo: swap an object while keeping everything else. Optionally add a reference image of the replacement.',
    kind: 'inplace',
    sourceLabel: 'Upload Image to Edit',
    sourceHint: 'Upload or paste (Ctrl/Cmd+V) the image you want to edit',
  },
  {
    id: 'restyle',
    label: 'Full restyle',
    prompt: 'Restyle this image as a moody editorial fashion photo while preserving the same person, face shape, hair, facial hair if present, pose, composition, and facial identity.',
    checkpoint: 'turbo',
    steps: 8,
    cfg: 1.0,
    groundingPx: 768,
    notes: 'Global style transfer over an existing photo, composition preserved. Pair with a moodboard style.',
    kind: 'inplace',
    sourceLabel: 'Upload Image to Restyle',
    sourceHint: 'Upload or paste (Ctrl/Cmd+V) the image to restyle',
  },
  {
    id: 'removal',
    label: 'Removal Raw mode',
    prompt: 'Remove the distracting object from the scene and naturally fill the background. Preserve the same person, face shape, hair, facial hair if present, and composition.',
    checkpoint: 'raw',
    steps: 20,
    cfg: 3.0,
    groundingPx: 768,
    notes: 'Remove objects from an existing photo. Model card recommends RAW at CFG 3 for removals and large deletions.',
    kind: 'inplace',
    sourceLabel: 'Upload Image to Edit',
    sourceHint: 'Upload or paste (Ctrl/Cmd+V) the image you want to edit',
  },
  {
    id: 'two_reference',
    label: 'Two people (place A + B)',
    prompt: 'A photo of two people together in the scene described in the placement boxes, natural interaction, consistent lighting.',
    checkpoint: 'turbo',
    steps: 8,
    cfg: 1.0,
    groundingPx: 1536,
    notes: 'Opens the placement boxes below: drag a box for each person and drop a face photo into each. Best with one face per box.',
    kind: 'compose',
    sourceLabel: '',
    sourceHint: '',
  },
]

export function presetById(id: CharacterEditTask): CharacterEditPreset {
  return characterEditPresets.find(preset => preset.id === id) ?? characterEditPresets[0]
}
