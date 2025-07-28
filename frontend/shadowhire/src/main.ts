import { AppModule } from './app/app.module';
import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';
import { enableProdMode } from '@angular/core';

// Enable for production
// enableProdMode();

platformBrowserDynamic().bootstrapModule(AppModule)
  .then(() => console.log('Application bootstrapped successfully'))
  .catch(err => {
    console.error('Bootstrap error:', err);
    // Add visual error feedback if needed
    document.body.innerHTML = '<h1>Application failed to load</h1><p>Check console for details</p>';
  });