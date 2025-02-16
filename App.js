import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // Import dark theme styles
import jobTransitions from "./jobTransitions"; // Import Job Role Transition Data

function App() {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expandedCourses, setExpandedCourses] = useState({});
  const [showJobs, setShowJobs] = useState(false);

  // Handle file selection
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  // Handle file upload and send to backend
  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file!");
      return;
    }

    setLoading(true);
    setShowJobs(false);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:8000/upload_resume/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResponse(res.data);
      setTimeout(() => setShowJobs(true), 500); // Add delay for smooth transition
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Failed to process file. Please try again!");
    } finally {
      setLoading(false);
    }
  };

  // Toggle "Read More" for skills display
  const toggleReadMore = (index) => {
    setExpandedCourses((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  return (
    <>
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="logo">Career Recommender</div>
        <ul className="nav-links">
          <li><a href="#">Home</a></li>
          <li><a href="#">About</a></li>
          <li><a href="#">Contact</a></li>
        </ul>
      </nav>

      {/* Main Content */}
      <div className="container">
        <h2>Resume-Based Career Recommendation</h2>

        {/* Upload Box */}
        <div className="upload-box">
          <input type="file" onChange={handleFileChange} accept=".pdf,.docx" />
          <button onClick={handleUpload} disabled={loading}>
            {loading ? "Uploading..." : "Upload Resume"}
          </button>
        </div>

        {loading && <p className="loading-text">Processing your resume...</p>}

        {/* Display Results */}
        {response && (
          <div className="result-box">
            <h3>Recommended Courses</h3>
            <div className="grid-container">
            {response.recommended_courses.map((course, index) => {
  const skillsList = course.Skills ? course.Skills.split(", ") : [];
  const displayedSkills = skillsList.slice(0, 5);
  const hiddenSkills = skillsList.slice(5);
  const isExpanded = expandedCourses[index];

  return (
    <div key={index} className="card">
      <h4>{course.Title}</h4>
      <p className="skills-covered">
        <strong>Skills Covered:</strong>
        <span className="skills-list">
          {displayedSkills.map((skill, i) => (
            <span key={i} className="skill">{skill}</span>
          ))}
          {isExpanded && hiddenSkills.map((skill, i) => (
            <span key={`extra-${i}`} className="skill">{skill}</span>
          ))}
        </span>
        {hiddenSkills.length > 0 && (
          <span className="read-more-text" onClick={() => toggleReadMore(index)}>
            {isExpanded ? " Show Less" : " ... Read More"}
          </span>
        )}
      </p>
      <a href={course.URL} target="_blank" rel="noopener noreferrer" className="view-btn">
        View Course
      </a>
    </div>
  );
})}


            </div>

            <h3>Recommended Jobs</h3>
            <div className={`grid-container-2-col ${showJobs ? "fade-in" : ""}`}>
              {response.recommended_jobs.map((job, index) => (
                <div key={index} className="card job-card">
                  <h4>{job.title}</h4>
                  <p>{job.description}</p>
                  
                  {/* Career Transition Tree for Each Job */}
                  <div className="career-tree">
  {jobTransitions[job.title]?.map((role, i) => (
    <div key={i} className="career-node">
      {role}
      {i < jobTransitions[job.title].length - 1 && <div className="career-connector"></div>}
    </div>
  )) || <span className="career-node">Career Path Not Available</span>}
</div>


                  {job.url && (
                    <a href={job.url} target="_blank" rel="noopener noreferrer" className="view-btn">
                      View Job Posting
                    </a>
                  )}
                </div>
              ))}
            </div>

          </div>
        )}
      </div>
    </>
  );
}

export default App;
