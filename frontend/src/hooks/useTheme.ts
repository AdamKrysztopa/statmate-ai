import { useEffect, useState } from 'react';

type Theme = 'light' | 'dark';

const storageKey = 'statmate-theme';

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>('dark');

  const applyTheme = (next: Theme) => {
    document.documentElement.setAttribute('data-theme', next);
    document.documentElement.classList.remove(next === 'dark' ? 'light' : 'dark');
    document.documentElement.classList.add(next === 'dark' ? 'dark' : 'light');
  };

  useEffect(() => {
    const stored = (localStorage.getItem(storageKey) as Theme | null) || undefined;
    if (stored) {
      setTheme(stored);
      applyTheme(stored);
      return;
    }
    const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
    const next = prefersLight ? 'light' : 'dark';
    setTheme(next);
    applyTheme(next);
  }, []);

  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem(storageKey, theme);
  }, [theme]);

  const toggle = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  return [theme, toggle];
}
