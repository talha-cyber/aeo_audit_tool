# German Localization Build Plan
## AEO Audit Tool - Multi-Language Support Implementation

### Overview
Implement German language support for the AEO audit tool, allowing clients to conduct audits and receive reports in German through a simple language toggle.

---

## Phase 1: Infrastructure & Configuration (2-3 days)

### 1.1 Language Configuration System
**Files to modify:**
- `app/core/platform_settings.py` - Add language settings
- `app/models/audit.py` - Add language field to audit configuration
- Database migration for language field

**Tasks:**
- [ ] Add `language` enum field to client configuration (`en`, `de`)
- [ ] Add language setting to audit run model
- [ ] Create database migration for new language fields
- [ ] Update API endpoints to accept language parameter
- [ ] Add language validation to configuration schema

### 1.2 Localization Infrastructure
**Files to create:**
- `app/core/i18n.py` - Internationalization utilities
- `locales/en.json` - English translations
- `locales/de.json` - German translations

**Tasks:**
- [ ] Create translation loading system
- [ ] Build translation key resolution utility
- [ ] Add language detection from client config
- [ ] Create translation helper functions
- [ ] Set up fallback mechanism (German → English)

---

## Phase 2: Question Generation Localization (3-4 days)

### 2.1 Question Template System
**Files to modify:**
- `app/services/question_generator.py` (if exists) or relevant question logic
- Identify where questions are generated/stored

**Tasks:**
- [ ] Audit current question generation system
- [ ] Create German question templates
- [ ] Implement template selection based on language
- [ ] Ensure question context includes language specification
- [ ] Test question generation in both languages

### 2.2 LLM Prompt Localization
**Files to modify:**
- `app/services/ai_platforms/*.py` - All platform clients
- Any prompt template files

**Tasks:**
- [ ] Identify all system prompts and instructions
- [ ] Create German versions of all prompts
- [ ] Add language context to LLM requests
- [ ] Update prompt templates to use localized text
- [ ] Ensure LLMs respond in requested language

---

## Phase 3: Analysis & Processing Localization (4-5 days)

### 3.1 NLP Pipeline German Support
**Files to modify:**
- `app/tasks/audit_tasks.py` - Main processing logic
- Any sentiment analysis components
- Requirements.txt - Add German NLP models

**Tasks:**
- [ ] Add German spaCy model (`de_core_news_sm`)
- [ ] Update sentiment analysis for German text
- [ ] Modify brand detection for German responses
- [ ] Test German text processing accuracy
- [ ] Benchmark performance with German models

### 3.2 Response Analysis Updates
**Files to modify:**
- Brand detection logic in audit processing
- Sentiment scoring mechanisms

**Tasks:**
- [ ] Ensure brand names are detected in German context
- [ ] Adapt sentiment analysis for German cultural context
- [ ] Update confidence scoring for German responses
- [ ] Create German-specific stop words if needed
- [ ] Validate analysis accuracy with German test data

---

## Phase 4: Report Generation Localization (5-6 days)

### 4.1 Report Content Translation
**Files to modify:**
- `app/services/report_generator.py` - All report sections
- Translation files for report content

**Tasks:**
- [ ] Translate all static report text (headers, labels, descriptions)
- [ ] Localize section titles and content structure
- [ ] Translate recommendation templates
- [ ] Update insight generation text
- [ ] Create German executive summary templates

### 4.2 Formatting & Cultural Adaptations
**Files to modify:**
- `app/services/report_generator.py` - Formatting logic

**Tasks:**
- [ ] Implement German date formatting (DD.MM.YYYY)
- [ ] Use German number formatting (1.234,56)
- [ ] Adapt currency formatting if applicable
- [ ] Update chart labels and legends to German
- [ ] Ensure proper German typography and spacing

### 4.3 PDF Styling for German
**Files to modify:**
- Report styling and layout logic

**Tasks:**
- [ ] Test German text length differences in layouts
- [ ] Adjust table column widths for German text
- [ ] Update PDF metadata to include language
- [ ] Ensure German special characters render correctly
- [ ] Test PDF generation with German content

---

## Phase 5: API & Frontend Integration (2-3 days)

### 5.1 API Updates
**Files to modify:**
- `app/api/v1/audits.py` - Audit endpoints
- API request/response models

**Tasks:**
- [ ] Add language parameter to audit creation
- [ ] Update API documentation for language support
- [ ] Add language validation to request schemas
- [ ] Update error messages for localization
- [ ] Test API with both language options

### 5.2 Frontend Language Toggle
**Files to modify:**
- Frontend client configuration components
- Audit setup forms

**Tasks:**
- [ ] Add language selection to client setup
- [ ] Create language toggle UI component
- [ ] Update form validation for language selection
- [ ] Store language preference in client configuration
- [ ] Add language indicator to audit runs

---

## Phase 6: Testing & Quality Assurance (3-4 days)

### 6.1 German Content Testing
**Test scenarios:**
- [ ] End-to-end audit in German
- [ ] German report generation and accuracy
- [ ] Mixed language environments
- [ ] Error handling in German
- [ ] Performance with German NLP models

### 6.2 Localization Quality Checks
**Tasks:**
- [ ] German translation review by native speaker
- [ ] Cultural appropriateness of recommendations
- [ ] Business terminology accuracy
- [ ] Consistency across all components
- [ ] User experience testing with German users

### 6.3 Regression Testing
**Tasks:**
- [ ] Ensure English functionality unchanged
- [ ] Test language switching scenarios
- [ ] Validate data integrity across languages
- [ ] Performance benchmarking
- [ ] Error handling in both languages

---

## Phase 7: Documentation & Deployment (2 days)

### 7.1 Documentation Updates
**Files to create/modify:**
- `docs/GERMAN_SUPPORT.md` - German localization guide
- API documentation updates
- User guide translations

**Tasks:**
- [ ] Document language configuration options
- [ ] Update API documentation for language parameters
- [ ] Create German user guide
- [ ] Document translation maintenance process
- [ ] Update deployment instructions

### 7.2 Deployment Preparation
**Tasks:**
- [ ] Update Docker configurations for German models
- [ ] Add German language models to deployment
- [ ] Update environment variables for localization
- [ ] Test deployment in staging environment
- [ ] Create rollback plan for language features

---

## Technical Requirements

### Dependencies to Add
```
# German NLP Support
spacy[de]==3.7.2
de_core_news_sm  # German spaCy model

# Enhanced i18n support (if needed)
babel==2.12.1
```

### Database Changes
```sql
-- Add language fields
ALTER TABLE audit_runs ADD COLUMN language VARCHAR(2) DEFAULT 'en';
ALTER TABLE clients ADD COLUMN preferred_language VARCHAR(2) DEFAULT 'en';
```

### Configuration Updates
```python
# app/core/settings.py
SUPPORTED_LANGUAGES = ["en", "de"]
DEFAULT_LANGUAGE = "en"
TRANSLATION_DIR = "locales"
```

---

## Effort Estimation

| Phase | Duration | Complexity | Dependencies |
|-------|----------|------------|--------------|
| Phase 1: Infrastructure | 2-3 days | Medium | None |
| Phase 2: Questions | 3-4 days | Medium | Phase 1 |
| Phase 3: Analysis | 4-5 days | High | Phase 1, 2 |
| Phase 4: Reports | 5-6 days | High | Phase 1, 3 |
| Phase 5: API/Frontend | 2-3 days | Low | Phase 1 |
| Phase 6: Testing | 3-4 days | Medium | All phases |
| Phase 7: Documentation | 2 days | Low | All phases |

**Total Estimated Duration: 21-27 days**

---

## Success Criteria

### Functional Requirements
- [x] Client can select German or English language
- [x] Questions generated and asked in selected language
- [x] LLM responses analyzed correctly in German
- [x] Reports generated in German with proper formatting
- [x] All static content properly translated

### Quality Requirements
- [x] German translation accuracy > 95%
- [x] No performance degradation > 10%
- [x] All existing English functionality preserved
- [x] Proper error handling in both languages
- [x] Cultural appropriateness of German content

### User Experience Requirements
- [x] Seamless language switching
- [x] Consistent language throughout audit process
- [x] Professional German business terminology
- [x] Proper German formatting conventions
- [x] Clear language selection interface

---

## Risk Assessment

### High Risk
- **German NLP accuracy**: May require fine-tuning for business context
- **Translation quality**: Business terminology requires domain expertise
- **Performance impact**: German models may increase processing time

### Medium Risk
- **Cultural adaptation**: Recommendations may need German market context
- **Text length variations**: German text often longer, may affect layouts
- **Testing complexity**: Requires German-speaking testers

### Low Risk
- **API integration**: Straightforward parameter addition
- **Database changes**: Simple schema additions
- **LLM multilingual support**: Already proven capability

---

## Maintenance Plan

### Ongoing Translation Management
- Process for updating German translations
- Version control for translation files
- Quality assurance for new features
- Regular German content review

### Performance Monitoring
- Monitor German processing performance
- Track German audit success rates
- Compare German vs English result quality
- User feedback collection system

---

## Step-by-Step Implementation Checklist
### Chronological Order with Engineering Tips

*From a time-traveling engineer who's already built this...*

### Day 1: Foundation Setup

#### 1. Environment Preparation
- [ ] **Create feature branch**: `git checkout -b feature/german-localization`
- [ ] **Backup database**: Always backup before schema changes
- [ ] **Install German spaCy model**: `python -m spacy download de_core_news_sm`

**🚨 Common Error**: Model download fails due to network/permissions
- **Fix**: Run with `--user` flag or in virtual environment
- **Fallback**: Download manually and install from file

#### 2. Database Schema Changes
- [ ] **Create migration**: `alembic revision -m "add_language_fields"`
- [ ] **Add language fields to models**: Start with audit_runs table
- [ ] **Test migration locally**: Always test before pushing

**🚨 Common Error**: Migration conflicts with existing data
- **Fix**: Use `op.add_column()` with default values
- **Pattern**: Always provide backwards-compatible defaults

#### 3. Core i18n Infrastructure
- [ ] **Create `app/core/i18n.py`**: Start simple, expand later
- [ ] **Create `locales/` directory structure**
- [ ] **Add basic translation loader**: JSON-based is easiest

**💡 Pro Tip**: Start with a simple key-value system. Don't over-engineer initially.

```python
# Simple pattern that works:
def get_text(key: str, lang: str = "en") -> str:
    return translations[lang].get(key, translations["en"].get(key, key))
```

### Day 2: Configuration & Settings

#### 4. Platform Settings Updates
- [ ] **Add language enum to `platform_settings.py`**
- [ ] **Update configuration validation**
- [ ] **Add language to audit run creation**

**🚨 Common Error**: Enum validation breaks existing configs
- **Fix**: Use Union types initially: `language: Union[str, Language]`
- **Migration**: Convert strings to enums in background task

#### 5. API Schema Updates
- [ ] **Update request schemas in `app/api/v1/audits.py`**
- [ ] **Add language validation**: Use pydantic validators
- [ ] **Test API endpoints with Postman/curl**

**💡 Pro Tip**: Add language parameter as optional first, make required later
```python
language: Optional[str] = "en"  # Start here
# Later: language: str = "en"
```

### Day 3: Question System Overhaul

#### 6. Audit Question Generation System
- [ ] **Find where questions are generated**: Usually in `audit_tasks.py` or separate service
- [ ] **Create question template system**: JSON files work well
- [ ] **Add language parameter to question functions**

**🚨 Common Error**: Hardcoded questions scattered throughout code
- **Pattern**: Search for all string literals containing "?"
- **Fix**: Extract to central question repository

#### 7. LLM Prompt Localization
- [ ] **Audit all AI platform clients**: Check every `.py` file in `ai_platforms/`
- [ ] **Extract system prompts**: Look for prompt templates
- [ ] **Add language context to LLM requests**

**💡 Pro Tip**: LLMs respond better when you explicitly state the language
```python
system_prompt = f"Please respond in {language_names[lang]}. {base_prompt}"
```

### Day 4: NLP Pipeline German Support

#### 8. German NLP Models Integration
- [ ] **Update `requirements.txt`**: Add spacy[de] and de_core_news_sm
- [ ] **Modify text processing in `audit_tasks.py`**
- [ ] **Add language-specific model loading**

**🚨 Common Error**: Memory issues loading multiple spaCy models
- **Fix**: Lazy load models, cache instances
- **Pattern**:
```python
@lru_cache(maxsize=None)
def get_nlp_model(lang: str):
    return spacy.load(f"{lang}_core_news_sm")
```

#### 9. Brand Detection Updates
- [ ] **Test brand detection with German text**
- [ ] **Update confidence thresholds**: German may need different values
- [ ] **Add German stop words if needed**

**💡 Pro Tip**: Brand names often transcend language barriers, but context matters

### Day 5-6: Report Generation Overhaul

#### 10. Translation File Creation
- [ ] **Extract all English text from `report_generator.py`**
- [ ] **Create comprehensive `locales/en.json`**
- [ ] **Generate `locales/de.json`** (use DeepL API for initial version)

**🚨 Common Error**: Missing translation keys cause crashes
- **Pattern**: Always provide fallback to English
- **Tool**: Use translation key validator in tests

#### 11. Report Content Localization
- [ ] **Update all section headers**: Use translation functions
- [ ] **Localize date/number formatting**
- [ ] **Update recommendation templates**

**💡 Pro Tip**: German business writing is more formal. Adjust tone accordingly.

#### 12. PDF Layout Adjustments
- [ ] **Test German text length**: Often 20-30% longer than English
- [ ] **Adjust table column widths**
- [ ] **Test special character rendering**: ä, ö, ü, ß

**🚨 Common Error**: German text overflows table cells
- **Fix**: Use dynamic column sizing or smaller fonts
- **Pattern**: Test with longest German business terms first

### Day 7: Integration & Testing

#### 13. End-to-End Integration
- [ ] **Connect all components**: Config → Questions → Analysis → Reports
- [ ] **Add language propagation**: Ensure language flows through entire pipeline
- [ ] **Test full audit run in German**

**🚨 Common Error**: Language gets lost between components
- **Fix**: Add language to all function signatures
- **Pattern**: Thread language through context objects

#### 14. Error Handling Localization
- [ ] **Localize error messages**
- [ ] **Update logging to include language context**
- [ ] **Test error scenarios in German**

### Day 8: Quality Assurance

#### 15. Automated Testing
- [ ] **Create German test data**: Sample questions, responses
- [ ] **Add language-specific unit tests**
- [ ] **Test language switching scenarios**

**💡 Pro Tip**: Create test fixtures with real German business scenarios

#### 16. Performance Testing
- [ ] **Benchmark German vs English processing**
- [ ] **Test memory usage with multiple models**
- [ ] **Check report generation times**

**🚨 Common Error**: German NLP models cause memory leaks
- **Fix**: Explicitly clean up model instances
- **Monitor**: Add memory usage logging

### Day 9: Polish & Documentation

#### 17. User Experience Polish
- [ ] **Test full user journey**
- [ ] **Verify German formatting everywhere**
- [ ] **Check PDF output quality**

#### 18. Documentation
- [ ] **Document language switching process**
- [ ] **Create troubleshooting guide**
- [ ] **Update API documentation**

### Day 10: Deployment Preparation

#### 19. Production Readiness
- [ ] **Update Docker configs**: Include German models
- [ ] **Add environment variables**
- [ ] **Create rollback plan**

#### 20. Final Integration Testing
- [ ] **Full regression test**: Ensure English still works
- [ ] **Load testing with German content**
- [ ] **Sign-off from German content reviewer**

---

## Engineering Patterns for Common Issues

### When Things Break (They Will)

#### **Pattern 1: Character Encoding Issues**
```python
# Always use UTF-8 explicitly
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
```

#### **Pattern 2: Translation Missing**
```python
def safe_translate(key: str, lang: str, fallback: str = None) -> str:
    try:
        return translations[lang][key]
    except KeyError:
        logger.warning(f"Missing translation: {key} for {lang}")
        return translations.get("en", {}).get(key, fallback or key)
```

#### **Pattern 3: Model Loading Failures**
```python
def load_nlp_model_safe(lang: str):
    try:
        return spacy.load(f"{lang}_core_news_sm")
    except OSError:
        logger.error(f"Could not load {lang} model, falling back to en")
        return spacy.load("en_core_web_sm")
```

#### **Pattern 4: PDF Layout Debugging**
```python
# Add debug mode for layout issues
if DEBUG_PDF_LAYOUT:
    story.append(Paragraph(f"DEBUG: Text length = {len(text)}", debug_style))
```

#### **Pattern 5: Database Migration Rollback**
```python
# Always test rollback scenarios
def downgrade():
    op.drop_column('audit_runs', 'language')
    # Test this path!
```

### Debugging Mindset

1. **Isolate the component**: Test each phase independently
2. **Use logging extensively**: Especially for language flow tracking
3. **Create minimal reproducible cases**: Single German question → Single German report
4. **Test boundaries**: Empty strings, special characters, very long German words
5. **Monitor resource usage**: German models are memory-intensive

### Success Indicators

- [ ] German audit completes without errors
- [ ] PDF report renders correctly with German text
- [ ] No memory leaks during language switching
- [ ] All translation keys have values
- [ ] Performance degradation < 10%
