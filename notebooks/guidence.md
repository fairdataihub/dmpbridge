# LLM-Based Document Structure Detection Strategy

## Goal

Detect and classify PDF narrative blocks into:

-   document_title
-   section
-   subsection
-   content

while preserving the original document structure and order.

## Core Principle

The system should determine:

> What role does this block play in the document hierarchy?

### Human-like reasoning

1.  Build document hierarchy
2.  Detect topic boundaries
3.  Use visual layout
4.  Determine semantic role
5.  Assign the best label

## Hierarchy First

Document ├── Title ├── Section │ └── Content ├── Section │ ├──
Subsection │ └── Content └── Section

## Strongest Signal

For every block:

-   Previous block
-   Current block
-   Next block

Ask:

-   Does the current block start a new idea?
-   Or continue the previous idea?

## Visual Signals

Use as supporting evidence:

-   Font size
-   Bold
-   Centering
-   Whitespace
-   Page position
-   Indentation

## Semantic Roles

### document_title

Main title of the document.

### section

Starts a major new topic.

### subsection

Prompt or internal heading under a section.

### content

Explanation, answer text, guidance, lists, paragraphs.

## Recommended DMPBridge Architecture

PDFPlumber extract text + layout ↓ Create previous/current/next context
↓ Render page image ↓ Qwen2-VL ↓ Label blocks ↓ Light postprocessing

## Key Philosophy

Do not ask:

Which regex pattern does this block match?

Ask:

What role does this block play in the document hierarchy?
