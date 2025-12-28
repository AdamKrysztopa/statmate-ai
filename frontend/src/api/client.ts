export type HealthResponse = { version?: string; status?: string };
export type User = { email: string };
export type Dataset = {
  id: string;
  original_filename: string;
  row_count?: number;
  created_at?: string;
  description?: string | null;
};
export type DatasetPreview = {
  dataset_id: string;
  original_filename: string;
  row_count: number;
  column_names: string[];
  preview_data: Record<string, unknown>[];
  data_types?: Record<string, string>;
  preview_rows?: number;
  description?: string | null;
};
export type AnalysisStatus = {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  message?: string;
  log_available?: boolean;
  version?: number;
  superseded_at?: string | null;
  comment?: string | null;
  decision_steps?: TraceStep[];
  intermediate_log?: string;
  execution_trace?: TraceStep[];
  assumption_log?: Record<string, unknown>[];
  progress?: number;
  workflow_graph?: WorkflowGraph;
};
export type TraceStep = {
  step: string;
  detail?: string;
  data?: Record<string, unknown>;
  p_value?: number;
  timestamp?: string;
  node?: string;
  node_id?: string;
  progress_pct?: number;
  step_index?: number;
  total_steps?: number;
};
export type TestHierarchyNode = {
  name: string;
  detail?: string;
  p_value?: number;
  assumptions?: Record<string, unknown>[];
  timestamp?: string;
  step_index?: number;
  total_steps?: number;
  node?: string;
};
export type TestHierarchy = {
  attempted?: TestHierarchyNode[];
  failures?: { test?: string; message?: string; timestamp?: string }[];
  chosen_test?: string | null;
  reviewer?: Record<string, unknown> | null;
};
export type PlotInfo = {
  title: string;
  description?: string;
  image_base64: string;
  column?: string;
  type?: string;
  content_type?: string;
};
export type WorkflowGraphAssets = {
  svg?: string;
  svg_base64?: string;
  png_base64?: string;
  alt?: string;
};
export type WorkflowGraph = {
  nodes: { id: string; label: string; kind: string; transitions: string[] }[];
  edges: { source: string; target: string; kind?: string }[];
  visited_nodes?: string[];
  selected_path?: string[];
  active_node?: string | null;
  chosen_test?: string | null;
  assets?: WorkflowGraphAssets;
};
export type ResultsDetail = {
  messages?: string[];
  probabilities?: Record<string, number>;
  effect_sizes?: Record<string, number>;
  summary?: string;
  full_output?: string;
  execution_trace?: TraceStep[];
  plots?: PlotInfo[];
  timestamp?: string;
  comment?: string | null;
  version?: number;
  test_hierarchy?: TestHierarchy;
  reviewer_report?: Record<string, unknown>;
  assumption_log?: Record<string, unknown>[];
  workflow_graph?: WorkflowGraph;
};
export type AnalysisResult = {
  id: string;
  status?: string;
  dataset_id?: string;
  dataset_name?: string;
  summary?: string;
  comment?: string | null;
  probabilities?: Record<string, number>;
  effect_sizes?: Record<string, number>;
  results_detail?: ResultsDetail;
  execution_trace?: TraceStep[];
  decision_steps?: TraceStep[];
  intermediate_log?: string;
  plots?: PlotInfo[];
  log_available?: boolean;
  model_name?: string;
  provider?: string;
  version?: number;
  superseded_at?: string | null;
  start_time?: string;
  end_time?: string;
  test_hierarchy?: TestHierarchy;
  reviewer_report?: Record<string, unknown>;
  assumption_log?: Record<string, unknown>[];
  workflow_graph?: WorkflowGraph;
};
export type AnalysisLog = { analysis_id: string; log_content: string; log_lines: string };
export type AnalysisListItem = {
  id: string;
  dataset_id: string;
  status: AnalysisStatus['status'];
  selected_columns?: string[];
  model_name?: string;
  provider?: string;
  start_time?: string;
  end_time?: string;
  version: number;
  superseded_at?: string | null;
  comment?: string | null;
  summary?: string | null;
};

export type AvailableModel = {
  name: string;
  provider: string;
  display_name: string;
  description?: string;
  capabilities?: string[];
  supports_tools?: boolean;
};

export type AvailableModelsResponse = {
  models: AvailableModel[];
  default_model: string;
  default_provider: string;
};

export class ApiClient {
  baseUrl: string;
  token?: string;

  constructor(baseUrl: string, token?: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.token = token;
  }

  withToken(token?: string) {
    return new ApiClient(this.baseUrl, token);
  }

  private headers(json = true): HeadersInit {
    const headers: HeadersInit = {};
    if (json) headers['Content-Type'] = 'application/json';
    if (this.token) headers['Authorization'] = `Bearer ${this.token}`;
    return headers;
  }

  async health(): Promise<HealthResponse> {
    const res = await fetch(`${this.baseUrl.replace(/\/api\/v1$/, '')}/health`);
    if (!res.ok) throw new Error('API unreachable');
    return res.json();
  }

  async login(email: string, password: string): Promise<{ access_token: string }> {
    const res = await fetch(`${this.baseUrl}/auth/login`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async register(email: string, password: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/auth/register`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) throw await this.error(res);
  }

  async me(): Promise<User> {
    const res = await fetch(`${this.baseUrl}/auth/me`, { headers: this.headers() });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async datasets(): Promise<Dataset[]> {
    const res = await fetch(`${this.baseUrl}/datasets/`, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async uploadDataset(file: File, description?: string): Promise<{ dataset_id: string }> {
    const form = new FormData();
    form.append('file', file);
    if (description) form.append('description', description);

    const res = await fetch(`${this.baseUrl}/datasets/upload`, {
      method: 'POST',
      headers: this.token ? { Authorization: `Bearer ${this.token}` } : undefined,
      body: form,
    });

    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async previewDataset(datasetId: string): Promise<DatasetPreview> {
    const res = await fetch(`${this.baseUrl}/datasets/${datasetId}/preview`, {
      headers: this.headers(false),
    });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async runAnalysis(args: {
    dataset_id: string;
    selected_columns?: string[];
    model_name?: string;
    provider?: string;
    overwrite?: boolean;
  }): Promise<{ id: string; version?: number }> {
    const res = await fetch(`${this.baseUrl}/analysis/run`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(args),
    });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async analysisStatus(id: string): Promise<AnalysisStatus> {
    const res = await fetch(`${this.baseUrl}/analysis/${id}`, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async analysisResults(id: string): Promise<AnalysisResult> {
    const res = await fetch(`${this.baseUrl}/analysis/${id}/results`, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async analysisLog(id: string): Promise<AnalysisLog> {
    const res = await fetch(`${this.baseUrl}/analysis/${id}/log`, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async exportAnalysis(id: string, format: 'pdf' | 'docx' | 'csv' | 'latex' | 'bundle'): Promise<Blob> {
    const res = await fetch(`${this.baseUrl}/analysis/${id}/export/${format}`, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.blob();
  }

  async analysisStream(id: string, signal?: AbortSignal): Promise<Response> {
    const headers = { ...this.headers(false), Accept: 'text/event-stream' };
    return fetch(`${this.baseUrl}/analysis/${id}/stream`, { headers, signal });
  }

  async workflowGraph(analysisId?: string): Promise<WorkflowGraph> {
    const url = analysisId
      ? `${this.baseUrl}/analysis/workflow-graph?analysis_id=${analysisId}`
      : `${this.baseUrl}/analysis/workflow-graph`;
    const res = await fetch(url, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async analyses(params: { dataset_id?: string; skip?: number; limit?: number } = {}): Promise<AnalysisListItem[]> {
    const query = new URLSearchParams();
    if (params.dataset_id) query.set('dataset_id', params.dataset_id);
    if (typeof params.skip === 'number') query.set('skip', String(params.skip));
    if (typeof params.limit === 'number') query.set('limit', String(params.limit));
    const qs = query.toString();
    const res = await fetch(`${this.baseUrl}/analysis/${qs ? `?${qs}` : ''}`, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async deleteAnalysis(id: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/analysis/${id}`, { method: 'DELETE', headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
  }

  async updateAnalysisComment(id: string, comment: string): Promise<AnalysisListItem> {
    const res = await fetch(`${this.baseUrl}/analysis/${id}/comment`, {
      method: 'PATCH',
      headers: this.headers(),
      body: JSON.stringify({ comment }),
    });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async renameColumns(datasetId: string, renames: Record<string, string>): Promise<DatasetPreview> {
    const res = await fetch(`${this.baseUrl}/datasets/${datasetId}/columns`, {
      method: 'PATCH',
      headers: this.headers(),
      body: JSON.stringify({ renames }),
    });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async updateDatasetDescription(datasetId: string, description: string | null): Promise<Dataset> {
    const res = await fetch(`${this.baseUrl}/datasets/${datasetId}/description`, {
      method: 'PATCH',
      headers: this.headers(),
      body: JSON.stringify({ description }),
    });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async availableModels(): Promise<AvailableModelsResponse> {
    const res = await fetch(`${this.baseUrl}/models/available`, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async configuredCredentials(): Promise<{
    configured_providers: string[];
    stored_credentials?: Record<string, string>;
    provider_quotas?: Record<string, number>;
  }> {
    const res = await fetch(`${this.baseUrl}/models/credentials`, { headers: this.headers(false) });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  async setCredentials(payload: {
    openai_api_key?: string;
    anthropic_api_key?: string;
    google_api_key?: string;
    groq_api_key?: string;
    gemini_api_key?: string;
    ollama_enabled?: string;
    ollama_base_url?: string;
    ollama_default_model?: string;
    openai_quota?: number | string;
    anthropic_quota?: number | string;
    google_quota?: number | string;
    gemini_quota?: number | string;
    groq_quota?: number | string;
  }): Promise<{ configured_providers: string[] }> {
    const res = await fetch(`${this.baseUrl}/models/credentials`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw await this.error(res);
    return res.json();
  }

  private async error(res: Response): Promise<Error> {
    const text = await res.text();
    let message = text || res.statusText;
    try {
      const data = JSON.parse(text);
      message = data.detail || data.message || message;
    } catch (_) {
      /* ignore */
    }
    return new Error(message || 'Request failed');
  }
}
