import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MlFeedbackPanelComponent } from './ml-feedback-panel.component';

describe('MlFeedbackPanelComponent', () => {
  let component: MlFeedbackPanelComponent;
  let fixture: ComponentFixture<MlFeedbackPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MlFeedbackPanelComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MlFeedbackPanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
