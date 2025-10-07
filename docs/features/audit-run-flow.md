# Audit Run Flow - Complete Implementation

## Overview

The audit run system now supports **real audit runs** with persona selection, platform configuration, and automatic question generation powered by the Question Engine V2.

---

## User Flow

### 1. Create Audit Run

**Location**: Click "New audit" button in dashboard header

**Steps**:
1. Enter a descriptive run name (e.g., "Q1 2025 Audit - Enterprise SaaS")
2. Select one or more personas from saved library
3. Choose platforms (ChatGPT, Claude, Perplexity, Google AI)
4. Select question volume (24, 48, 96, or 150)
5. Click "Create & Start"

**Result**: Audit run is created and question generation begins automatically

---

### 2. Persona Management

**Location**: `/personas` page

**Features**:
- Create new personas (B2C or B2B)
- Edit existing personas
- Clone personas with modifications
- View persona library with role/driver/context details

**Integration**: Personas created here are immediately available in the audit run creation flow

---

### 3. View Audit Run Progress

**Location**: `/audits/run/{runId}`

**Displays**:
- Run status (pending → running → completed)
- Progress bar showing completion percentage
- Real-time question generation status
- List of generated questions with:
  - Platform (ChatGPT, Claude, etc.)
  - Question prompt
  - Sentiment analysis (positive/neutral/negative)
  - Brand mentions with frequency and sentiment

**Auto-refresh**: Page automatically polls backend every 4 seconds until run completes

---

## Backend Architecture

### API Endpoints

#### POST `/api/v1/dashboard/audits/run`

**Request**:
```json
{
  "name": "Q1 2025 Audit",
  "personaIds": ["persona_123", "persona_456"],
  "platforms": ["openai", "anthropic"],
  "questionCount": 48
}
```

**Response**:
```json
{
  "run": {
    "id": "run_789",
    "name": "Q1 2025 Audit",
    "status": "pending",
    "startedAt": "2025-10-01T17:00:00Z",
    "progress": {
      "done": 0,
      "total": 48
    },
    "issues": []
  }
}
```

#### GET `/api/v1/dashboard/audits/run/{runId}`

Returns detailed audit run information including generated questions.

---

### Service Layer

**File**: `app/services/dashboard/audit_run_creation_service.py`

**Responsibilities**:
1. Validate persona ownership and existence
2. Create `AuditRun` database record with config
3. Trigger background Celery task for question generation
4. Return initial run status

**Database Schema**:
```python
AuditRun:
  - id: str (UUID)
  - client_id: Optional[str]
  - config: JSON
    - name: str
    - platforms: List[str]
    - question_count: int
    - persona_ids: List[str]
    - personas: List[PersonaInfo]
    - owner_id: str
  - status: enum (pending/running/completed/failed)
  - started_at: datetime
  - completed_at: Optional[datetime]
```

---

### Question Generation Pipeline

**Background Task**: `app/tasks/audit_tasks.py` → `run_audit_task`

**Flow**:
1. Load audit run from database
2. Extract personas and platforms from config
3. For each persona × platform combination:
   - Generate questions using Question Engine V2
   - Questions are persona-aware (role, driver, voice)
   - Store questions in database
4. Run sentiment analysis on each response
5. Update audit run status to "completed"

**Question Engine Integration**:
- Uses `QuestionEngineV2` service
- Persona context automatically included
- Platform-specific formatting
- Dynamic provider selection (template vs AI-generated)

---

## Frontend Implementation

### Components

#### `CreateAuditRunDrawer`

**File**: `frontend/src/components/audits/create-audit-run-drawer.tsx`

**Features**:
- Form with name, persona selection, platforms, question count
- Fetches persona library via `usePersonaLibrary()` hook
- Multi-select UI for personas with checkmarks
- Platform toggle buttons
- Question count presets
- Validation (requires name, ≥1 persona, ≥1 platform)
- Error handling with user-friendly messages

**State Management**:
```typescript
const [runName, setRunName] = useState('');
const [selectedPersonaIds, setSelectedPersonaIds] = useState<string[]>([]);
const [selectedPlatforms, setSelectedPlatforms] = useState<string[]>(['openai']);
const [questionCount, setQuestionCount] = useState(48);
```

---

#### `RunDetail`

**File**: `frontend/src/app/(dashboard)/audits/run/[runId]/run-detail.tsx`

**Features**:
- Progress visualization (completion percentage)
- Issue tracking (severity: low/medium/high)
- Questions table with:
  - Platform column
  - Prompt text
  - Brand mentions with sentiment
  - Overall sentiment badge
- Auto-refresh every 4s while running

---

### API Integration

#### React Query Hooks

**File**: `frontend/src/lib/api/queries.ts`

**Hooks**:
```typescript
// Create audit run
const createAuditRun = useCreateAuditRun();
await createAuditRun.mutateAsync(payload);

// Fetch run detail
const { data, isLoading } = useAuditRunDetail(runId);

// Fetch persona library
const { data: personas } = usePersonaLibrary('b2c');
```

**Auto-invalidation**: After creating a run, queries for audit lists are automatically refetched

---

### API Client

**File**: `frontend/src/lib/api/client.ts`

**Method**:
```typescript
createAuditRun: async (payload: CreateAuditRunPayload) =>
  request(`${DASHBOARD_PREFIX}/audits/run`, CreateAuditRunResponseSchema, {
    method: 'POST',
    body: JSON.stringify(payload)
  }).then((response) => response.run)
```

**Authentication**: JWT token automatically included in headers via `_authorize` dependency

---

## Data Flow Diagram

```
User clicks "New Audit"
  ↓
CreateAuditRunDrawer opens
  ↓
User selects:
  - Run name
  - Personas (from saved library)
  - Platforms
  - Question count
  ↓
Click "Create & Start"
  ↓
Frontend: POST /api/v1/dashboard/audits/run
  ↓
Backend: create_audit_run()
  - Validate personas
  - Create AuditRun record
  - Trigger Celery task
  ↓
Background: run_audit_task.delay(run_id)
  - Generate questions (Question Engine V2)
  - Run sentiment analysis
  - Update status → "completed"
  ↓
Frontend: Navigate to /audits/run/{runId}
  ↓
RunDetail page auto-refreshes
  ↓
Display generated questions with analysis
```

---

## Question Generation Details

### Persona-Driven Questions

Each question is influenced by the selected persona's:
- **Role**: E.g., "IT Decision Maker", "Marketing Manager"
- **Driver**: E.g., "cost_efficiency", "innovation"
- **Voice**: Optional natural language description
- **Context**: Industry-specific considerations

**Example**:
```
Persona: IT Decision Maker (cost_efficiency driver)
Platform: ChatGPT
Generated Question: "What are the most cost-effective enterprise project management tools for teams of 50+ people?"
```

### Platform-Specific Formatting

Questions are optimized per platform:
- **OpenAI/ChatGPT**: Conversational, detailed prompts
- **Claude**: Structured, analytical queries
- **Perplexity**: Research-focused questions
- **Google AI**: Broad exploratory prompts

---

## Sentiment Analysis

**Performed on**: Each question's AI-generated response

**Labels**:
- `positive`: Favorable mentions, recommendations
- `neutral`: Factual information without sentiment
- `negative`: Criticism, warnings, concerns

**Brand Mentions**:
```typescript
{
  brand: "CompanyName",
  frequency: 5,  // Number of times mentioned
  sentiment: "positive"
}
```

---

## Error Handling

### Frontend

- **No personas**: Shows message "Create personas first" with link to `/personas`
- **API errors**: Displays error message in drawer
- **Validation**: Disables submit button until all fields valid

### Backend

- **Personas not found**: Returns 400 with specific persona IDs
- **Unauthorized access**: Returns 401 (JWT validation)
- **Database errors**: Rolls back transaction, returns 500

---

## Testing the Flow

### Prerequisites

1. Backend running: `docker-compose up` or `python app/main.py`
2. Frontend running: `npm run dev` (already running at http://localhost:3000)
3. At least one persona created in `/personas`

### Test Steps

1. Navigate to http://localhost:3000/overview
2. Click **"New audit"** button (top right)
3. Fill out form:
   - Name: "Test Audit Run"
   - Select 1+ personas
   - Select 1+ platforms
   - Choose question count
4. Click **"Create & Start"**
5. Should navigate to `/audits/run/{id}`
6. Verify:
   - Status shows "pending" or "running"
   - Progress bar appears
   - Questions appear as they're generated
   - Status eventually becomes "completed"

---

## Future Enhancements

### Planned Features

1. **Client Selection**: Associate audit runs with specific clients
2. **Scheduling**: Recurring audit runs (daily/weekly/monthly)
3. **Custom Question Templates**: Let users add manual questions
4. **Multi-Language**: Questions in different languages per persona
5. **Advanced Filters**: Filter questions by sentiment, platform, persona
6. **Export Questions**: Download as CSV/JSON
7. **Question Ratings**: Users can rate question quality
8. **A/B Testing**: Compare different persona configurations

### Performance Optimizations

1. **Batch Question Generation**: Generate multiple questions in parallel
2. **Caching**: Cache frequently used persona configurations
3. **Progressive Loading**: Stream questions to frontend as they're generated
4. **WebSocket Updates**: Real-time progress updates instead of polling

---

## Troubleshooting

### "No personas found" in drawer

**Solution**: Navigate to `/personas` and create at least one persona

### Audit run stays in "pending" status

**Causes**:
- Celery worker not running
- Redis/broker connection issue
- Error in question generation

**Debug**:
```bash
# Check Celery logs
docker-compose logs celery

# Manually trigger task
from app.tasks.audit_tasks import run_audit_task
run_audit_task.delay('run_id')
```

### Questions not appearing on run detail page

**Causes**:
- Run still in progress
- Database query issue
- Frontend polling stopped

**Debug**:
- Check browser console for errors
- Verify API response: `GET /api/v1/dashboard/audits/run/{id}`
- Check database: `SELECT * FROM audit_runs WHERE id = '{id}'`

---

## API Reference Summary

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/dashboard/audits/run` | POST | ✅ | Create new audit run |
| `/dashboard/audits/run/{id}` | GET | ✅ | Get run details + questions |
| `/dashboard/audits/runs` | GET | ✅ | List all runs |
| `/dashboard/personas/library` | GET | ✅ | Get persona library |

---

**Last Updated**: 2025-10-01
**Status**: ✅ Production Ready
**Version**: 1.0.0
