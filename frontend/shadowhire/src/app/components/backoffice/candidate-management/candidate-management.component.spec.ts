import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CandidateManagementComponent } from './candidate-management.component';

describe('CandidateManagementComponent', () => {
  let component: CandidateManagementComponent;
  let fixture: ComponentFixture<CandidateManagementComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CandidateManagementComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CandidateManagementComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
