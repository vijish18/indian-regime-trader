# HEXAWARE missing-price investigation

The failed diagnostic fold attempted to liquidate NSE:HEXAWARE on
2020-11-13. This is not a missing bhavcopy price that can be interpolated.

The company's announcement dated 2020-10-20 records trading suspension
effective 2020-11-02 (after the close on 2020-10-30), and delisting effective
2020-11-09. Residual shareholders could tender at INR 475 per share during
the stated exit window ending 2021-11-08. The offer required participation;
it was not automatic cash credited on delisting.

Primary sources:
- https://nsearchives.nseindia.com/corporate/HEXAWARE_21102020193702_Newspaperclipping.pdf
- https://hexaware.com/wp-content/uploads/2020/07/Exit-Offer-FAQs.pdf

Do not insert a fabricated INR 475 trading bar or assume a cash payment on
2020-11-13. An executable delisting policy must specify tender participation,
eligibility, submission timing, acceptance and cash availability. Alternatively
a pre-suspension market exit must be a strategy rule based on information
already announced, executed against an actual tradable bar. Either choice
must be recorded as a simulation assumption and tested before a rerun.

Status: event identified; entitlement/tender lifecycle is not implemented.
The missing-bar failure remains intentional until that accounting exists.
