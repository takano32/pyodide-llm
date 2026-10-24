// What a worker of the runtime has that TypeScript's WebWorker library does not say (T.368.3). Read by tsc alone
// (tsconfig.worker.json): nothing of this file is in what a browser gets.

// navigator.deviceMemory: Chromium's, the device's memory in GB as a power of two (8 at most on Android); other
// browsers do not say it
interface WorkerNavigator {
  readonly deviceMemory?: number;
}
