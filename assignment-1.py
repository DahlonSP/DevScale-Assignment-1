import os

from dotenv import load_dotenv
from openai import BaseModel, OpenAI

load_dotenv()

SUMOPOD_API_KEY = os.getenv("SUMOPOD_API_KEY")
SUMOPOD_BASE_URL = os.getenv("SUMOPOD_BASE_URL")

client = OpenAI(base_url=SUMOPOD_BASE_URL, api_key=SUMOPOD_API_KEY)

class AirworthinessDirective (BaseModel):
    ad_no : str
    effective_date : str
    affected_ads: str
    unsafe_condition: str

completion = client.chat.completions.parse(
    model="nvidia/nemotron-3-nano-30b",
    messages=[
        {
            "role": "system",
            "content": "Extract the following information from the given text and return it in a structured format: AD No., Effective Date, Affected ADs, Unsafe Condition.",
        },
        {
            "role": "user",
            "content": f""" AD NO: 2026-09-01 The Boeing Company: Amendment 39-23321; Docket No. FAA-2025-1114; Project Identifier AD-2025-00314-T.
                        (a) Effective Date

                        This airworthiness directive (AD) is effective June 3, 2026.
                        (b) Affected ADs

                        This AD replaces AD 2023-08-04, Amendment 39-22419 (88 FR 33823, May 25, 2023) (AD 2023-08-04).
                        (c) Applicability

                        This AD applies to The Boeing Company Model 787-8, 787-9, and 787-10 airplanes, certificated in any category, as specified in Boeing Alert Requirements Bulletin B787-81205-SB380021-00 RB, Issue 001, dated August 12, 2022.
                        (d) Subject

                        Air Transport Association (ATA) of America Code 38, Water/waste.
                        (e) Unsafe Condition

                        This AD was prompted by reports of a loss of water pressure during flight and water leaks that affected multiple pieces of electronic equipment, and by the determination that some clamshell couplings for certain lavatory and galley doors did not have a required safety strap. The FAA is issuing this AD to prevent the unsafe condition, which, if not addressed, could lead to water leaks and water migration to critical flight equipment, which may affect the continued safe flight and landing of the airplane. """
        }    ],
    response_format=AirworthinessDirective
)

final_output = completion.choices[0].message.parsed
assert final_output is not None

print(final_output.model_dump())