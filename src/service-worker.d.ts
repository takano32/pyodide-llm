// What public/coi.js stands on, for its type check alone (tsconfig.sw.json, T.368.1). coi.js is a classic script that
// is served as it is, and not a character of it is written for the checker but its first line: so what a Service
// Worker's `self` has that any worker's does not is said here, where TypeScript's own types say it of
// ServiceWorkerGlobalScope and type `self` as a WorkerGlobalScope. (A declaration file: nothing of it is served.)
interface WorkerGlobalScope {
  skipWaiting(): Promise<void>;
  readonly clients: Clients;
  addEventListener<K extends keyof ServiceWorkerGlobalScopeEventMap>(type: K,
    listener: (this: ServiceWorkerGlobalScope, event: ServiceWorkerGlobalScopeEventMap[K]) => unknown, options?: boolean | AddEventListenerOptions): void;
}
