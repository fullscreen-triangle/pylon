import { useState, useEffect } from "react";

export function useThemeSwitch() {
  const [mode, setMode] = useState("dark");

  useEffect(() => {
    document.documentElement.classList.add("dark");
    window.localStorage.setItem("theme", "dark");
  }, []);

  // mode is always dark — setter is a no-op kept for API compatibility
  return [mode, () => {}];
}
