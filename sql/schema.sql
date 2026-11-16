-- Manual MySQL 8+ setup. The app also creates these tables on startup.
-- Run as a MySQL admin: mysql -u root -p < sql/schema.sql
-- Change the password below and put the same one in DATABASE_URL in .env.
CREATE DATABASE IF NOT EXISTS aftercare CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'aftercare'@'localhost' IDENTIFIED BY 'change_me';
GRANT ALL PRIVILEGES ON aftercare.* TO 'aftercare'@'localhost';
FLUSH PRIVILEGES;

USE aftercare;

CREATE TABLE IF NOT EXISTS users (
  id INT NOT NULL AUTO_INCREMENT,
  username VARCHAR(50) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) NOT NULL DEFAULT 'patient',
  display_name VARCHAR(100) NULL,
  specialty VARCHAR(100) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY ix_users_username (username)
);

CREATE TABLE IF NOT EXISTS discharge_documents (
  id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'pending_review',  -- pending_review | confirmed | archived
  filename VARCHAR(255) NULL,
  text LONGTEXT NOT NULL,
  discharge_date DATE NULL,
  anchor_date DATE NULL,
  date_source VARCHAR(20) NULL,                           -- document | confirmation
  unverified JSON NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  confirmed_at DATETIME NULL,
  active_user_id INT NULL,                                -- = user_id while pending_review/confirmed, NULL when archived
  PRIMARY KEY (id),
  UNIQUE KEY uq_discharge_documents_active_user (active_user_id),
  KEY ix_discharge_documents_user_id (user_id),
  FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document_files (             -- the uploaded PDF, shown beside the instructions
  document_id INT NOT NULL,
  content_type VARCHAR(100) NOT NULL,
  data LONGBLOB NOT NULL,
  PRIMARY KEY (document_id),
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS extracted_items (
  id INT NOT NULL AUTO_INCREMENT,
  document_id INT NOT NULL,
  category VARCHAR(50) NOT NULL,
  text TEXT NOT NULL,
  source_quote TEXT NOT NULL,
  details JSON NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'active',           -- active | rejected
  PRIMARY KEY (id),
  KEY ix_extracted_items_document_id (document_id),
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checklist_items (
  id INT NOT NULL AUTO_INCREMENT,
  document_id INT NOT NULL,
  extracted_item_id INT NULL,
  label VARCHAR(500) NOT NULL,
  slot VARCHAR(20) NOT NULL DEFAULT 'anytime',            -- morning | midday | evening | bedtime | anytime
  start_date DATE NOT NULL,
  end_date DATE NULL,
  one_time BOOLEAN NOT NULL DEFAULT FALSE,
  approximate BOOLEAN NOT NULL DEFAULT FALSE,
  undated BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (id),
  KEY ix_checklist_items_document_id (document_id),
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE CASCADE,
  FOREIGN KEY (extracted_item_id) REFERENCES extracted_items (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS checklist_completions (
  id INT NOT NULL AUTO_INCREMENT,
  checklist_item_id INT NOT NULL,
  day DATE NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_completion_item_day (checklist_item_id, day),
  FOREIGN KEY (checklist_item_id) REFERENCES checklist_items (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id INT NOT NULL AUTO_INCREMENT,
  document_id INT NOT NULL,
  role VARCHAR(20) NOT NULL,                              -- user | assistant
  content TEXT NOT NULL,
  payload JSON NULL,
  doctor_id INT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY ix_chat_messages_document_id (document_id),
  KEY ix_chat_messages_doctor_id (doctor_id),
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE CASCADE,
  FOREIGN KEY (doctor_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS care_team_questions (
  id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  document_id INT NULL,
  question TEXT NOT NULL,
  reason TEXT NOT NULL,
  source VARCHAR(30) NOT NULL,                            -- agent | rejected_item | validation_fallback
  status VARCHAR(20) NOT NULL DEFAULT 'open',             -- open | answered | resolved
  answer TEXT NULL,
  answered_by INT NULL,
  answered_at DATETIME NULL,
  doctor_id INT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY ix_care_team_questions_user_id (user_id),
  KEY ix_care_team_questions_doctor_id (doctor_id),
  FOREIGN KEY (doctor_id) REFERENCES users (id) ON DELETE SET NULL,
  FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
  FOREIGN KEY (answered_by) REFERENCES users (id) ON DELETE SET NULL,
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS doctor_guidance (
  id INT NOT NULL AUTO_INCREMENT,
  doctor_id INT NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  scope VARCHAR(20) NOT NULL,                             -- general | patient
  patient_id INT NULL,
  document_id INT NULL,
  source_question_id INT NULL,
  drug_tags JSON NOT NULL,
  condition_tags JSON NOT NULL,
  active TINYINT(1) NOT NULL DEFAULT 1,
  embedding JSON NULL,
  embedding_model VARCHAR(100) NULL,
  rules_status VARCHAR(20) NULL,                          -- NULL (not tried) | ready | failed
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY ix_doctor_guidance_doctor_id (doctor_id),
  KEY ix_doctor_guidance_patient_id (patient_id),
  FOREIGN KEY (doctor_id) REFERENCES users (id) ON DELETE CASCADE,
  FOREIGN KEY (patient_id) REFERENCES users (id) ON DELETE CASCADE,
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE SET NULL,
  FOREIGN KEY (source_question_id) REFERENCES care_team_questions (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS guidance_rules (
  id INT NOT NULL AUTO_INCREMENT,
  guidance_id INT NOT NULL,
  doctor_id INT NOT NULL,
  position INT NOT NULL,
  statement TEXT NOT NULL,                                -- standalone restatement
  quote TEXT NOT NULL,                                    -- exact span of the answer
  embedding JSON NULL,
  embedding_model VARCHAR(100) NULL,
  kind VARCHAR(20) NOT NULL DEFAULT 'advice',             -- advice | habit
  habit_type VARCHAR(20) NULL,                            -- ask | close | warning
  warning_terms JSON NULL,                                -- NULL reads as []
  suppressed TINYINT(1) NOT NULL DEFAULT 0,               -- a doctor-deleted habit
  PRIMARY KEY (id),
  KEY ix_guidance_rules_guidance_id (guidance_id),
  KEY ix_guidance_rules_doctor_id (doctor_id),
  FOREIGN KEY (guidance_id) REFERENCES doctor_guidance (id) ON DELETE CASCADE,
  FOREIGN KEY (doctor_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS guidance_conflicts (
  id INT NOT NULL AUTO_INCREMENT,
  doctor_id INT NOT NULL,
  rule_id INT NOT NULL,
  other_rule_id INT NULL,
  label_rxcui VARCHAR(20) NULL,
  label_title VARCHAR(500) NULL,
  label_url VARCHAR(500) NULL,
  label_quote TEXT NULL,
  explanation TEXT NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'open',             -- open | dismissed
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY ix_guidance_conflicts_doctor_id (doctor_id),
  KEY ix_guidance_conflicts_rule_id (rule_id),
  KEY ix_guidance_conflicts_other_rule_id (other_rule_id),
  FOREIGN KEY (doctor_id) REFERENCES users (id) ON DELETE CASCADE,
  FOREIGN KEY (rule_id) REFERENCES guidance_rules (id) ON DELETE CASCADE,
  FOREIGN KEY (other_rule_id) REFERENCES guidance_rules (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS answer_drafts (
  id INT NOT NULL AUTO_INCREMENT,
  question_id INT NOT NULL,
  doctor_id INT NOT NULL,
  sentences JSON NOT NULL,                                -- [{text, sources, supported, problem}]
  evidence JSON NOT NULL,                                 -- [{number, kind, title, quote}]
  text TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  sent_answer TEXT NULL,
  kept DOUBLE NULL,                                       -- 0..1, share of the draft kept
  sent_unchanged TINYINT(1) NULL,
  sent_at DATETIME NULL,
  PRIMARY KEY (id),
  KEY ix_answer_drafts_question_id (question_id),
  KEY ix_answer_drafts_doctor_id (doctor_id),
  FOREIGN KEY (question_id) REFERENCES care_team_questions (id) ON DELETE CASCADE,
  FOREIGN KEY (doctor_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS drug_labels (
  id INT NOT NULL AUTO_INCREMENT,
  rxcui VARCHAR(20) NOT NULL,
  ingredient VARCHAR(200) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'fetching',         -- fetching | pending | approved | rejected | no_label | failed
  candidates JSON NOT NULL,
  setid VARCHAR(64) NULL,
  title VARCHAR(500) NULL,
  url VARCHAR(500) NULL,
  text LONGTEXT NULL,
  sha256 VARCHAR(64) NULL,
  collected_at DATETIME NULL,
  error TEXT NULL,
  reviewed_by INT NULL,
  reviewed_at DATETIME NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_drug_labels_rxcui (rxcui),
  FOREIGN KEY (reviewed_by) REFERENCES users (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS document_drugs (
  id INT NOT NULL AUTO_INCREMENT,
  document_id INT NOT NULL,
  extracted_item_id INT NOT NULL,
  name VARCHAR(200) NOT NULL,
  rxcui VARCHAR(20) NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'fetching',         -- fetching | resolved | unresolved | failed
  error TEXT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_document_drugs_item_rxcui (extracted_item_id, rxcui),
  KEY ix_document_drugs_document_id (document_id),
  KEY ix_document_drugs_extracted_item_id (extracted_item_id),
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE CASCADE,
  FOREIGN KEY (extracted_item_id) REFERENCES extracted_items (id) ON DELETE CASCADE
);

-- New attachment table only: existing question/document tables are not altered.
CREATE TABLE IF NOT EXISTS voice_clips (
  id INT NOT NULL AUTO_INCREMENT,
  question_id INT NOT NULL,
  sender_id INT NOT NULL,
  kind VARCHAR(20) NOT NULL,
  storage_key VARCHAR(40) NOT NULL,
  duration_ms INT NOT NULL,
  byte_size INT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_voice_question_kind (question_id, kind),
  UNIQUE KEY uq_voice_storage_key (storage_key),
  KEY ix_voice_clips_question_id (question_id),
  FOREIGN KEY (question_id) REFERENCES care_team_questions (id) ON DELETE CASCADE,
  FOREIGN KEY (sender_id) REFERENCES users (id) ON DELETE CASCADE
);
