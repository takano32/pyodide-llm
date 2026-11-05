// What the page's scripts read that no module declares: Vite's import.meta.env, of which the pages use the site's base
// alone. (T368.1. A declaration file: nothing of it is in a build. __BUILD__ is declared by each module that reads it.)
interface ImportMetaEnv {
  readonly BASE_URL: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
