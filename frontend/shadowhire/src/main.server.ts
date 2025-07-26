// src/main.server.ts

import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';

const bootstrap = () => bootstrapApplication(AppComponent);

export default bootstrap;  // <-- this is the default export
