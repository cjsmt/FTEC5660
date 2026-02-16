from __future__ import annotations

from typing import TypedDict, List, Optional, Dict, Any


class WorkExperience(TypedDict, total=False):
    """ Work history """
    company: Optional[str]
    title: Optional[str]
    seniority: Optional[str]
    start_year: Optional[int]
    end_year: Optional[int]
    is_current: Optional[bool]


class Education(TypedDict, total=False):
    """ Academic history """
    school: Optional[str]
    degree: Optional[str]
    field: Optional[str]
    start_year: Optional[int]
    end_year: Optional[int]


class Skill(TypedDict, total=False):
    name: Optional[str]
    proficiency: Optional[int]


class LinkedInProfile(TypedDict, total=False):
    """ LinkedIn professional profile including work history, education, and skills. """
    id: Optional[int]
    name: Optional[str]
    headline: Optional[str]
    city: Optional[str]
    country: Optional[str]
    industry: Optional[str]
    status: Optional[str]
    years_experience: Optional[int]
    summary: Optional[str]
    skills: List[Skill]
    experience: List[WorkExperience]
    education: List[Education]


class FacebookProfile(TypedDict, total=False):
    """ Facebook profile including personal info, bio, relationships, and activity. """
    id: Optional[int]
    display_name: Optional[str]
    original_name: Optional[str]
    city: Optional[str]
    country: Optional[str]
    hometown: Optional[str]
    bio: Optional[str]
    status: Optional[str]
    education: Optional[str]
    current_job: Optional[str]
    current_company: Optional[str]
    interests: Optional[str]
    friends: List[int]
    posts: List[Dict[str, Any]]


class ResumeData(TypedDict, total=False):
    """ Resume data extracted from PDF """
    name: Optional[str]
    city: Optional[str]
    country: Optional[str]
    hometown: Optional[str]
    headline: Optional[str]
    skills: List[Skill]
    experience: List[WorkExperience]
    education: List[Education]
    raw_text: Optional[str]


class ExperienceComparison(TypedDict, total=False):
    """ Experience comparison """
    score: float
    common_experience: List[WorkExperience]
    only_in_resume: List[WorkExperience]
    only_in_social: List[WorkExperience]
    summary: str
    details: List[str]


class EducationComparison(TypedDict, total=False):
    """ Education comparison """
    score: float
    resume_education: List[Education]
    social_education: List[Education]
    summary: str
    details: List[str]


class SkillsComparison(TypedDict, total=False):
    """ Skills comparison """
    score: float
    common_skills: List[Skill]
    only_in_resume: List[Skill]
    only_in_social: List[Skill]
    summary: str
    details: List[str]


class VerificationReport(TypedDict, total=False):
    """ Final verification report """
    resume: ResumeData
    linkedin_profile: Optional[LinkedInProfile]
    facebook_profile: Optional[FacebookProfile]
    skills_comparison: Optional[SkillsComparison]
    experience_comparison: Optional[ExperienceComparison]
    education_comparison: Optional[EducationComparison]
    summary: str


class CVState(TypedDict, total=False):
    """State structure used in LangGraph"""
    resume_path: str
    resume_data: Optional[ResumeData]
    linkedin_profile: Optional[LinkedInProfile]
    facebook_profile: Optional[FacebookProfile]
    report: Optional[VerificationReport]
