export type HealthResponse = { version?: string; status?: string };
export type User = { email: string };
export type Dataset = { id: string; original_filename: string; row_count?: number; created_at?: string };
export type DatasetPreview = {
  dataset_id: string;
  original_filename: string;
  row_count: number;
  column_names: string[];
  preview_data: Record<string, unknown>[];
};
export type AnalysisStatus = {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message?: string;
  log_available?: boolean;
};
export type TraceStep = { step: string; detail?: string; data?: Record<string, unknown>; p_value?: number; timestamp?: string };
export type PlotInfo = {
  title: string;
  description?: string;
  image_base64: string;
  column?: string;
  type?: string;
  content_type?: string;
};
export type ResultsDetail = {
  messages?: string[];
  probabilities?: Record<string, number>;
  summary?: string;
  full_output?: string;
  execution_trace?: TraceStep[];
  plots?: PlotInfo[];
  timestamp?: string;
};
export type AnalysisResult = {
  id: string;
  status?: string;
  dataset_id?: string;
  dataset_name?: string;
  summary?: string;
  probabilities?: Record<string, number>;
  results_detail?: ResultsDetail;
  execution_trace?: TraceStep[];
  plots?: PlotInfo[];
  log_available?: boolean;
  model_name?: string;
  provider?: string;
};
export type AnalysisLog = { analysis_id: string; log_content: string; log_lines: string };

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

  async runAnalysis(args: { dataset_id: string; selected_columns?: string[]; model_name?: string; provider?: string }): Promise<{ id: string }> {
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
