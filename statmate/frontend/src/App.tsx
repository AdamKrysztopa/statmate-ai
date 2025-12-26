import { FormEvent, useEffect, useMemo, useState } from 'react';
import { ApiClient, AnalysisResult, AnalysisStatus, DatasetPreview, Dataset, User } from './api/client';
import { useTheme } from './hooks/useTheme';

const defaultApi = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1';

function App() {
  const [theme, toggleTheme] = useTheme();
  const [apiBase, setApiBase] = useState(defaultApi);
  const [token, setToken] = useState<string | undefined>(undefined);
  const [user, setUser] = useState<User | undefined>();
  const [health, setHealth] = useState<string>('');
  const [error, setError] = useState<string>('');

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [preview, setPreview] = useState<DatasetPreview | undefined>();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [registering, setRegistering] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [description, setDescription] = useState('');
  const [file, setFile] = useState<File | null>(null);

  const [analysisId, setAnalysisId] = useState<string>('');
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus | undefined>();
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | undefined>();

  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [modelName, setModelName] = useState('gpt-4o');
  const [provider, setProvider] = useState('openai');

  const api = useMemo(() => new ApiClient(apiBase, token), [apiBase, token]);

  const clearMessages = () => {
    setError('');
    setHealth('');
  };

  const connect = async () => {
    clearMessages();
    try {
      const info = await api.health();
      setHealth(info.version ? `API online (v${info.version})` : 'API online');
    } catch (e) {
      setError((e as Error).message || 'Unable to reach API');
    }
  };

  const handleLogin = async (evt: FormEvent) => {
    evt.preventDefault();
    clearMessages();
    try {
      const res = await api.login(email, password);
      setToken(res.access_token);
      const me = await api.withToken(res.access_token).me();
      setUser(me);
      await loadDatasets(api.withToken(res.access_token));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleRegister = async (evt: FormEvent) => {
    evt.preventDefault();
    clearMessages();
    try {
      await api.register(email, password);
      setRegistering(false);
      setHealth('Account created. You can log in now.');
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const loadDatasets = async (client = api) => {
    if (!token && client.token === undefined) return;
    try {
      const list = await client.datasets();
      setDatasets(list);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const selectDataset = async (id: string) => {
    setSelectedDataset(id);
    setAnalysisResults(undefined);
    setAnalysisStatus(undefined);
    setAnalysisId('');
    if (!id) return;
    try {
      const p = await api.previewDataset(id);
      setPreview(p);
      setSelectedColumns([]);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    clearMessages();
    try {
      const result = await api.uploadDataset(file, description || undefined);
      setHealth(`Uploaded dataset ${result.dataset_id.slice(0, 8)}…`);
      setDescription('');
      setFile(null);
      await loadDatasets();
      await selectDataset(result.dataset_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedDataset) {
      setError('Pick a dataset first');
      return;
    }
    clearMessages();
    try {
      const { id } = await api.runAnalysis({
        dataset_id: selectedDataset,
        selected_columns: selectedColumns.length ? selectedColumns : undefined,
        model_name: modelName || undefined,
        provider: provider || undefined,
      });
      setAnalysisId(id);
      setAnalysisStatus({ id, status: 'pending' });
      setAnalysisResults(undefined);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const refreshStatus = async () => {
    if (!analysisId) return;
    try {
      const status = await api.analysisStatus(analysisId);
      setAnalysisStatus(status);
      if (status.status === 'completed') {
        const results = await api.analysisResults(analysisId);
        setAnalysisResults(results);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  };

  useEffect(() => {
    const storedToken = localStorage.getItem('statmate-token');
    if (storedToken) {
      setToken(storedToken);
    }
  }, []);

  useEffect(() => {
    if (token) {
      localStorage.setItem('statmate-token', token);
      api.withToken(token)
        .me()
        .then(setUser)
        .catch(() => setToken(undefined));
      loadDatasets(api.withToken(token));
    } else {
      localStorage.removeItem('statmate-token');
      setUser(undefined);
      setDatasets([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    if (analysisId) {
      const interval = setInterval(refreshStatus, 4000);
      return () => clearInterval(interval);
    }
    return undefined;
  }, [analysisId]);

  return (
    <div className="app-shell">
      <header className="header">
        <div className="logo">
          <div className="logo-mark">Σ</div>
          <div className="title-block">
            <h1>StatmateAI Frontend</h1>
            <p>Modern React client for FastAPI + LangGraph</p>
          </div>
        </div>
        <div className="controls">
          <button className="button" onClick={toggleTheme} aria-label="Toggle theme">
            {theme === 'dark' ? '🌙 Dark' : '☀️ Light'}
          </button>
          <button className="button" onClick={connect}>
            🔌 Check API
          </button>
        </div>
      </header>

      {error && <div className="toast error">{error}</div>}
      {health && <div className="toast">{health}</div>}

      <section className="card-grid">
        <div className="card">
          <h3>API Connection</h3>
          <p className="muted">Point to your running FastAPI instance.</p>
          <div className="input-group">
            <label>API Base URL</label>
            <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="http://localhost:8000/api/v1" />
          </div>
          <div className="pill-row">
            <span className="badge">ENV: {import.meta.env.MODE}</span>
            <span className="badge">Theme: {theme}</span>
            <span className="badge">{user ? `Signed in: ${user.email}` : 'Anon mode'}</span>
          </div>
        </div>

        <div className="card">
          <h3>{registering ? 'Create Account' : 'Sign In'}</h3>
          <form onSubmit={registering ? handleRegister : handleLogin}>
            <div className="input-group">
              <label>Email</label>
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
            </div>
            <div className="input-group">
              <label>Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
            </div>
            <div className="pill-row">
              <button className="button primary" type="submit">
                {registering ? 'Create & Login' : 'Login'}
              </button>
              <button className="button" type="button" onClick={() => setRegistering((v) => !v)}>
                {registering ? 'Have an account? Sign in' : 'Need an account? Register'}
              </button>
              {token && (
                <button className="button" type="button" onClick={() => setToken(undefined)}>
                  Log out
                </button>
              )}
            </div>
          </form>
        </div>

        <div className="card">
          <h3>Upload Dataset</h3>
          <div className="input-group">
            <label>File</label>
            <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </div>
          <div className="input-group">
            <label>Description (optional)</label>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} placeholder="Study notes or variables" />
          </div>
          <button className="button primary" disabled={!file || uploading || !token} onClick={handleUpload}>
            {uploading ? 'Uploading…' : 'Upload'}
          </button>
          {!token && <p className="muted">Sign in to upload.</p>}
        </div>
      </section>

      <div className="section-title">Datasets</div>
      <div className="card">
        <div className="pill-row" style={{ marginBottom: 12 }}>
          <span className="badge">{datasets.length} available</span>
          <button className="button" disabled={!token} onClick={() => loadDatasets()}>
            Refresh list
          </button>
        </div>
        <div className="pill-row">
          {datasets.map((ds) => (
            <button
              key={ds.id}
              className="button"
              style={{
                borderColor: selectedDataset === ds.id ? 'var(--primary)' : 'var(--border)',
                color: selectedDataset === ds.id ? 'var(--primary)' : 'var(--text)',
              }}
              onClick={() => selectDataset(ds.id)}
            >
              {selectedDataset === ds.id ? '✓ ' : ''}
              {ds.original_filename}
            </button>
          ))}
          {!datasets.length && <span className="muted">No datasets yet.</span>}
        </div>

        {preview && (
          <div style={{ marginTop: 16 }}>
            <div className="pill-row">
              <span className="badge">Rows: {preview.row_count}</span>
              <span className="badge">Columns: {preview.column_names.length}</span>
            </div>
            <div className="input-group" style={{ marginTop: 12 }}>
              <label>Select columns for analysis (optional)</label>
              <div className="pill-row">
                {preview.column_names.map((name) => {
                  const active = selectedColumns.includes(name);
                  return (
                    <button
                      key={name}
                      className="button"
                      style={{ borderColor: active ? 'var(--primary)' : 'var(--border)', color: active ? 'var(--primary)' : 'var(--text)' }}
                      type="button"
                      onClick={() =>
                        setSelectedColumns((cols) => (cols.includes(name) ? cols.filter((c) => c !== name) : [...cols, name]))
                      }
                    >
                      {active ? '✓ ' : ''}
                      {name}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="table-like">
              <table>
                <thead>
                  <tr>
                    {preview.column_names.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.preview_data.slice(0, 5).map((row, idx) => (
                    <tr key={idx}>
                      {preview.column_names.map((c) => (
                        <td key={c}>{String((row as Record<string, unknown>)[c])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      <div className="section-title">Analysis</div>
      <div className="card-grid">
        <div className="card">
          <h3>Model</h3>
          <div className="input-group">
            <label>Provider</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="google">Gemini</option>
              <option value="groq">Groq</option>
              <option value="ollama">Ollama</option>
            </select>
          </div>
          <div className="input-group">
            <label>Model name</label>
            <input value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder="gpt-4o" />
          </div>
          <button className="button primary" disabled={!selectedDataset || !token} onClick={handleRunAnalysis}>
            🚀 Run analysis
          </button>
          {!selectedDataset && <p className="muted">Pick a dataset first.</p>}
        </div>

        <div className="card">
          <h3>Status</h3>
          <div className="input-group">
            <label>Analysis ID</label>
            <input value={analysisId} onChange={(e) => setAnalysisId(e.target.value)} placeholder="auto-filled after run" />
          </div>
          <div className="pill-row">
            <button className="button" disabled={!analysisId} onClick={refreshStatus}>
              Refresh
            </button>
            <span className="badge">{analysisStatus ? `Status: ${analysisStatus.status}` : 'Waiting to start'}</span>
          </div>
          {analysisStatus?.message && <p className="muted">{analysisStatus.message}</p>}
        </div>
      </div>

      {analysisResults && (
        <div className="card" style={{ marginTop: 12 }}>
          <h3>Results</h3>
          <div className="pill-row" style={{ marginBottom: 12 }}>
            {analysisResults.model_name && <span className="tag">Model: {analysisResults.model_name}</span>}
            {analysisResults.provider && <span className="tag">Provider: {analysisResults.provider}</span>}
          </div>
          {analysisResults.summary && <p>{analysisResults.summary}</p>}
          {analysisResults.probabilities && (
            <div className="table-like" style={{ marginTop: 12 }}>
              <table>
                <thead>
                  <tr>
                    <th>Test</th>
                    <th>P-Value</th>
                    <th>Significance</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(analysisResults.probabilities).map(([name, value]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{value.toFixed(4)}</td>
                      <td>{value < 0.05 ? 'Significant' : 'Not significant'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {analysisResults.results_detail && (
            <pre className="card" style={{ overflow: 'auto', background: 'var(--bg-input)' }}>
{JSON.stringify(analysisResults.results_detail, null, 2)}
            </pre>
          )}
        </div>
      )}

      <footer className="footer">
        <span>StatmateAI • React + Vite • FastAPI backend</span>
      </footer>
    </div>
  );
}

export default App;
